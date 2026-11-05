// Generate training data for WiezenLM (hand evaluator).
// For random deals and every seat: the outcome of each contract this seat could play, computed with
// the bot's own rollout (heuristic early tricks + exact double-dummy endgame) on the true deal.
//   node train/gen_bidding.js <deals> <seed> <out.bin>
// record (25 bytes): 13 cards, turned card, relative seat (0 = dealer, 1 = voorhand),
//   tricks alone in ♣ ♦ ♥ ♠ (declarer leads), tricks solo no-trump, tricks "alleen" (turned trump,
//   voorhand leads), misery clean (0/1), team tricks with partner at offset +1, +2, +3 (turned trump).
"use strict";
const fs = require("fs");
const R = require("../engine/rules.js");
const { rollout, ddSetup } = require("../engine/ai.js");
const { DDS } = require("../engine/dds.js");

const [deals, seed, out] = [+process.argv[2], +process.argv[3], process.argv[4]];
const rand = R.rngFrom(seed);
const dds = new DDS(19);
const buf = Buffer.alloc(deals * 4 * 25);
let off = 0;
const t0 = Date.now();
for (let d = 0; d < deals; d++) {
  const deck = R.shuffleInPlace([...Array(52).keys()], rand);
  const dealer = Math.floor(rand() * 4);
  const pattern = rand() < 0.5 ? "445" : "454";
  const map = R.dealMap(dealer, pattern);
  const hands = [[], [], [], []];
  deck.forEach((c, i) => hands[map[i]].push(c));
  const turned = deck[51], ts = R.suitOf(turned), voor = (dealer + 1) % 4;
  for (let seat = 0; seat < 4; seat++) {
    const run = (ct, leader) => rollout({ hands: hands.map((h) => h.slice()), trick: [], leader, turn: leader, total: 0, ct, forced: null }, ddSetup(ct, seat), dds);
    const lab = [];
    for (let t = 0; t < 4; t++) lab.push(run({ type: "abondance", players: [seat], trump: t, level: 9 }, seat));
    lab.push(run({ type: "solo", players: [seat], trump: R.NT }, seat));
    lab.push(run({ type: "alleen", players: [seat], trump: ts }, voor));
    lab.push(run({ type: "miserie", players: [seat], trump: R.NT }, voor) === 0 ? 1 : 0);
    for (let k = 1; k <= 3; k++) lab.push(run({ type: "vraag", players: [seat, (seat + k) % 4], trump: ts }, voor));
    hands[seat].forEach((c, i) => buf.writeUInt8(c, off + i));
    buf.writeUInt8(turned, off + 13);
    buf.writeUInt8((seat - dealer + 4) % 4, off + 14);
    lab.forEach((v, i) => buf.writeUInt8(v, off + 15 + i));
    off += 25;
  }
  if ((d + 1) % 500 === 0) console.error(`${d + 1}/${deals} deals, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
fs.writeFileSync(out, buf.subarray(0, off));
console.error("done", off / 25, "samples");
