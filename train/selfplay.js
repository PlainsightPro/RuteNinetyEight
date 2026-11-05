// Self-play data for WiezenLM v2.
//   node train/selfplay.js <bots|fast> <hands> <seed> <out.bin> [wiezenlm.web.json]
// Sessions of 24 hands without shuffling (habits, card tracking). For every deal and seat we store:
//   - the hand, turned card, seat relative to the dealer
//   - the seat's card-location belief at the start of the hand (tracking + learned habits)
//   - counterfactual labels: tricks for 10 contracts by rollout on the true deal (as gen_bidding.js)
//   - in "bots" mode: the REAL outcome of the contract the bots actually played (bot vs bot)
// record (184 bytes): 13 cards | turned | rel | 156 loc bytes (P*255, card-major, seats +1..+3) |
//   10 rollout labels | realHead (255 = none) | realValue | mode | spare
"use strict";
const fs = require("fs");
const R = require("../engine/rules.js");
const { Bot, rollout, ddSetup, locFeatures, makeBelief, knownMap, heurCard, voidsOf, sampleHands } = require("../engine/ai.js");
const { Habits } = require("../engine/belief.js");
const { DDS } = require("../engine/dds.js");
const { WiezenLM } = require("../engine/wiezenlm.js");

const [mode, handsArg, seed, out, wpath] = process.argv.slice(2);
const N = +handsArg, rand = R.rngFrom(+seed), dds = new DDS(19);
const REC = 184;
let prior = null;
if (wpath) {
  const lm = new WiezenLM(JSON.parse(fs.readFileSync(wpath, "utf8")));
  prior = (h, t, s, loc) => lm.predict(h, t, s, loc);
  prior.usesLoc = !!lm.usesLoc;
}
const bot = new Bot({ rand, prior, worlds: 14, bidWorlds: 8, timeMs: 0 });

// quick rule-based players (only used to create many realistic tracking situations)
function rulesBid(g, seat) {
  const h = g.hands[seat], ts = g.turnedSuit, legal = g.legalBids(seat), has = (t) => legal.find((b) => b.type === t);
  const trumps = h.filter((c) => R.suitOf(c) === ts).length, honours = h.filter((c) => R.rankOf(c) >= 10).length;
  if (g.askAlone === seat) return trumps >= 6 && honours >= 4 ? { type: "alleen" } : { type: "pas" };
  if (has("mee") && trumps >= 3 && honours >= 3) return { type: "mee" };
  if (has("vraag") && trumps >= 4 && honours >= 3) return { type: "vraag" };
  return { type: "pas" };
}
function rulesPlay(g, seat) {
  const legal = g.legalCards(seat);
  if (legal.length === 1) return legal[0];
  const b = makeBelief(g, seat, false); b.update(knownMap(g, seat));
  const c = heurCard({ hands: sampleHands(g, seat, b, rand, voidsOf(g)), trick: g.trick.slice(), leader: g.leader, ct: g.contract }, seat);
  return legal.includes(c) ? c : legal[0];
}

// which network head a played contract corresponds to, from `seat`'s point of view
function realHead(ct, seat, won, turnedSuit) {
  if (!ct || !ct.players.includes(seat)) return null;
  const t = ct.players.reduce((a, p) => a + won[p], 0);
  switch (ct.type) {
    case "vraag": { const o = ct.players.find((p) => p !== seat); return { head: 6 + ((o - seat + 4) % 4), value: t }; }
    case "alleen": return { head: 5, value: t };
    case "abondance": return { head: ct.trump, value: t };
    case "solo": return { head: ct.trump === R.NT ? 4 : ct.trump, value: t };
    case "soloslim": return { head: turnedSuit, value: t };
    case "miserie": case "openmiserie": return { head: 10, value: won[seat] === 0 ? 1 : 0 }; // misery clean
  }
  return null; // troel / troela: not a head of the network
}

const buf = Buffer.alloc(N * 4 * REC);
let off = 0, hands = 0, real = 0;
const t0 = Date.now();
while (hands < N) {
  const g = new R.Game({ rand, dealer: Math.floor(rand() * 4) });
  const H = new Habits();
  let seen = 0;
  for (let k = 0; k < 24 && hands < N; k++, hands++) {
    g.deal();
    const ts = g.turnedSuit, voor = (g.dealer + 1) % 4;
    const recs = [];
    for (let seat = 0; seat < 4; seat++) {
      const B = makeBelief(g, seat, true, H);
      B.update(knownMap(g, seat));
      const loc = locFeatures(g, seat, B, rand, 64);
      const run = (ct, leader) => rollout({ hands: g.dealt.map((h) => h.slice()), trick: [], leader, turn: leader, total: 0, ct, forced: null }, ddSetup(ct, seat), dds);
      const lab = [];
      for (let t = 0; t < 4; t++) lab.push(run({ type: "abondance", players: [seat], trump: t, level: 9 }, seat));
      lab.push(run({ type: "solo", players: [seat], trump: R.NT }, seat));
      lab.push(run({ type: "alleen", players: [seat], trump: ts }, voor));
      lab.push(run({ type: "miserie", players: [seat], trump: R.NT }, voor) === 0 ? 1 : 0);
      for (let o = 1; o <= 3; o++) lab.push(run({ type: "vraag", players: [seat, (seat + o) % 4], trump: ts }, voor));
      recs.push({ seat, loc, lab });
    }
    if (mode === "bots") {
      while (g.phase === "bid") g.bid(g.turn, bot.chooseBid(g, g.turn).bid);
      while (g.phase === "play") g.play(g.turn, bot.choosePlay(g, g.turn).card);
    } else {
      while (g.phase === "bid") g.bid(g.turn, rulesBid(g, g.turn));
      while (g.phase === "play") g.play(g.turn, rulesPlay(g, g.turn));
    }
    const ct = g.contract, last = g.history[g.history.length - 1];
    for (const r of recs) {
      const dealt = last.dealt[r.seat];
      dealt.forEach((c, i) => buf.writeUInt8(c, off + i));
      buf.writeUInt8(g.turned, off + 13); // still the turned card of the hand just played
      buf.writeUInt8((r.seat - last.dealer + 4) % 4, off + 14);
      for (let i = 0; i < 156; i++) buf.writeUInt8(Math.round(Math.min(1, Math.max(0, r.loc[i])) * 255), off + 15 + i);
      r.lab.forEach((v, i) => buf.writeUInt8(v, off + 171 + i));
      const rh = mode === "bots" ? realHead(ct, r.seat, last.won, ts) : null;
      buf.writeUInt8(rh ? rh.head : 255, off + 181);
      buf.writeUInt8(rh ? rh.value : 0, off + 182);
      buf.writeUInt8(mode === "bots" ? 1 : 0, off + 183);
      if (rh) real++;
      off += REC;
    }
    for (; seen < g.history.length; seen++) H.learnFrom(g.history[seen]);
    if ((hands + 1) % 20 === 0) {
      console.error(`${hands + 1}/${N} hands, ${real} real labels, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
      fs.writeFileSync(out, buf.subarray(0, off)); // checkpoint
    }
  }
}
fs.writeFileSync(out, buf.subarray(0, off));
console.error("done", off / REC, "samples,", real, "real labels");
