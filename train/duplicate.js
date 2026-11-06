// Duplicate-format match: every deal is played twice with the same cards, the two bot types
// swapping seats, so card luck largely cancels out (as in duplicate bridge).
//   node train/duplicate.js <deals> <seed> <A> <B>
// bot types: v1 | v3 | v3inf (v3 + bidding inference) | v3deep (v3 + exact endgame from 8 cards, more deals)
// Decks are shuffled every deal here (duplicate needs identical decks), so card tracking plays no role.
"use strict";
const fs = require("fs");
const path = require("path");
const R = require("../engine/rules.js");
const { Bot } = require("../engine/ai.js");
const { WiezenLM } = require("../engine/wiezenlm.js");

const [deals, seed, A, B] = [+process.argv[2], +process.argv[3], process.argv[4], process.argv[5]];
const root = path.join(__dirname, "..");
function load(p) {
  const lm = new WiezenLM(JSON.parse(fs.readFileSync(path.join(root, p), "utf8")));
  const f = (h, t, s, loc) => lm.predict(h, t, s, loc);
  f.usesLoc = lm.usesLoc;
  return f;
}
const base = { worlds: 20, bidWorlds: 10, timeMs: 0 };
const make = (name, rand) => {
  switch (name) {
    case "v1": return new Bot({ ...base, rand, prior: load("model/v1/wiezenlm.web.json") });
    case "v3": return new Bot({ ...base, rand, prior: load("model/wiezenlm.web.json") });
    case "v3inf": return new Bot({ ...base, rand, prior: load("model/wiezenlm.web.json"), bidModel: load("model/bid/wiezenlm.web.json") });
    case "v3deep": return new Bot({ ...base, rand, prior: load("model/wiezenlm.web.json"), kdd: 8, worlds: 30 });
  }
  throw new Error("unknown bot " + name);
};
const bots = [make(A, R.rngFrom(seed * 7 + 1)), make(B, R.rngFrom(seed * 7 + 2))];

let n = 0, sum = 0, sq = 0, differ = 0;
for (let d = 0; d < deals; d++) {
  const dealSeed = seed * 1000003 + d;
  const diffs = [], ctr = [];
  for (const pass of [0, 1]) {
    const typeOf = (p) => ((p % 2 === pass) ? 0 : 1); // pass 0: A in seats 0+2, pass 1: A in seats 1+3
    const G = new R.Game({ rand: R.rngFrom(dealSeed), dealer: d % 4, alwaysShuffle: true, habits: false });
    G.deal();
    while (G.phase === "bid") G.bid(G.turn, bots[typeOf(G.turn)].chooseBid(G, G.turn).bid);
    while (G.phase === "play") G.play(G.turn, bots[typeOf(G.turn)].choosePlay(G, G.turn).card);
    const avg = [0, 0];
    for (let p = 0; p < 4; p++) avg[typeOf(p)] += G.lastPts[p] / 2;
    diffs.push(avg[0] - avg[1]);
    ctr.push(G.contract ? G.contract.type + ":" + G.contract.players.join("") : "pas");
  }
  if (ctr[0] !== ctr[1]) differ++;
  const x = (diffs[0] + diffs[1]) / 2;
  n++; sum += x; sq += x * x;
  if (n % 10 === 0) {
    const m = sum / n, se = Math.sqrt(Math.max(0, sq / n - m * m) / n);
    console.error(`${n}/${deals} deals: ${A} - ${B} = ${m.toFixed(2)} ± ${(1.96 * se).toFixed(2)} points per player per hand (95%), contract differed in ${differ}`);
  }
}
const m = sum / n, se = Math.sqrt(Math.max(0, sq / n - m * m) / n);
console.log(JSON.stringify({ A, B, deals: n, diff: +m.toFixed(3), ci95: +(1.96 * se).toFixed(3), contract_differed: differ }));
