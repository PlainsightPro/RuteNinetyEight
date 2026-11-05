// Strength evaluation: two player types share the table (seats 0+2 vs 1+3, swapped every hand).
// Contracts form partnerships dynamically, so we report average points per hand per player type.
//   node train/match.js <hands> <seed> <matchup> [wiezenlm.web.json]
// matchups: bot-vs-rules | tracking-vs-notracking | net-vs-net (bots with model A vs model B: pass both paths)
"use strict";
const R = require("../engine/rules.js");
const { Bot, heurCard, makeBelief, knownMap, voidsOf, sampleHands } = require("../engine/ai.js");
const { WiezenLM } = require("../engine/wiezenlm.js");

const [hands, seed, matchup, wpath, wpath2] = [+process.argv[2], +process.argv[3], process.argv[4], process.argv[5], process.argv[6]];
const rand = R.rngFrom(seed);
let prior = null;
if (wpath) { const lm = new WiezenLM(JSON.parse(require("fs").readFileSync(wpath, "utf8"))); prior = (h, t, s, loc) => lm.predict(h, t, s, loc); prior.usesLoc = lm.usesLoc; }

// A sensible rule-based club player: bids on trump length/honours, plays the heuristic on one guessed deal.
class RulesPlayer {
  chooseBid(g, seat) {
    const h = g.hands[seat], ts = g.turnedSuit, legal = g.legalBids(seat);
    const has = (t) => legal.find((b) => b.type === t);
    const trumps = h.filter((c) => R.suitOf(c) === ts), aces = h.filter((c) => R.rankOf(c) === 12).length;
    const honours = h.filter((c) => R.rankOf(c) >= 10).length;
    const lows = h.filter((c) => R.rankOf(c) <= 6).length, high = h.filter((c) => R.rankOf(c) >= 11).length;
    if (g.askAlone === seat) return { bid: trumps.length >= 6 && honours >= 4 ? { type: "alleen" } : { type: "pas" } };
    for (let t = 0; t < 4; t++) {
      const tr = h.filter((c) => R.suitOf(c) === t);
      if (tr.length >= 7 && aces >= 2) { const b = legal.find((x) => x.type === "abondance" && x.level === 9 && x.trump === t); if (b) return { bid: b }; }
    }
    if (lows >= 10 && high === 0 && has("miserie")) return { bid: { type: "miserie" } };
    if (has("mee") && trumps.length >= 3 && honours >= 3) return { bid: { type: "mee" } };
    if (has("vraag") && trumps.length >= 4 && honours >= 3) return { bid: { type: "vraag" } };
    return { bid: { type: "pas" } };
  }
  choosePlay(g, seat) {
    const legal = g.legalCards(seat);
    if (legal.length === 1) return { card: legal[0] };
    const b = makeBelief(g, seat, false);
    b.update(knownMap(g, seat));
    const hands = sampleHands(g, seat, b, rand, voidsOf(g));
    const st = { hands, trick: g.trick.slice(), leader: g.leader, ct: g.contract, forced: null };
    const c = heurCard(st, seat);
    return { card: legal.includes(c) ? c : legal[0] };
  }
}

function loadPrior(p) { const lm = new WiezenLM(JSON.parse(require("fs").readFileSync(p, "utf8"))); const f = (h, t, s, loc) => lm.predict(h, t, s, loc); f.usesLoc = lm.usesLoc; return f; }
const mk = (opts) => new Bot({ rand, prior, worlds: 20, bidWorlds: 10, timeMs: 0, ...opts });
const [X, Y, names] = matchup === "tracking-vs-notracking" ? [mk({}), mk({ useTracking: false }), ["tracking", "no-tracking"]]
  : matchup === "net-vs-net" ? [mk({ prior: loadPrior(wpath) }), mk({ prior: loadPrior(wpath2) }), ["modelA", "modelB"]]
  : [mk({}), new RulesPlayer(), ["bot", "rules"]];

const G = new R.Game({ rand, dealer: 0 });
const tot = [0, 0], cnt = [0, 0], contracts = {};
let dsum = 0, dsq = 0; // per-hand difference (avg points of type X seats - type Y seats)
let declared = [0, 0], made = [0, 0];
for (let h = 0; h < hands; h++) {
  const typeOf = (p) => ((p + h) % 2 === 0 ? 0 : 1); // seats alternate X/Y, shifted every hand
  const player = (p) => (typeOf(p) === 0 ? X : Y);
  G.deal();
  while (G.phase === "bid") G.bid(G.turn, player(G.turn).chooseBid(G, G.turn).bid);
  while (G.phase === "play") G.play(G.turn, player(G.turn).choosePlay(G, G.turn).card);
  const k = G.contract ? G.contract.type : "allpass";
  contracts[k] = (contracts[k] || 0) + 1;
  if (G.contract) {
    const ty = typeOf(G.contract.players[0]);
    declared[ty]++;
    if (G.lastPts[G.contract.players[0]] > 0) made[ty]++;
  }
  const hp = [0, 0];
  for (let p = 0; p < 4; p++) { tot[typeOf(p)] += G.lastPts[p]; cnt[typeOf(p)]++; hp[typeOf(p)] += G.lastPts[p] / 2; }
  dsum += hp[0] - hp[1]; dsq += (hp[0] - hp[1]) ** 2;
  if ((h + 1) % 10 === 0) {
    const n = h + 1, m = dsum / n, se = Math.sqrt(Math.max(0, dsq / n - m * m) / n);
    console.error(`${n}/${hands} hands: ${names[0]} - ${names[1]} = ${m.toFixed(2)} ± ${(1.96 * se).toFixed(2)} points per player per hand (95%)`);
  }
}
const per = tot.map((t, i) => t / cnt[i]);
console.log(JSON.stringify({ matchup, hands, [names[0]]: +per[0].toFixed(3), [names[1]]: +per[1].toFixed(3),
  diff_per_hand: +(per[0] - per[1]).toFixed(3), declared: { [names[0]]: declared[0], [names[1]]: declared[1] },
  made: { [names[0]]: made[0], [names[1]]: made[1] }, contracts }));
