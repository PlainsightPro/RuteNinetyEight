// Engine tests: node train/test_engine.js
"use strict";
const assert = require("assert");
const R = require("../engine/rules.js");
const { DDS } = require("../engine/dds.js");
const { Belief } = require("../engine/belief.js");

let failures = 0;
function test(name, fn) {
  try { fn(); console.log("ok  ", name); } catch (e) { failures++; console.log("FAIL", name, "\n     ", e.message); }
}

// ---------------------------------------------------------------- scoring vs. the IWWA table
const pts = (ct, tricks) => R.scoreHand(ct, tricks);
test("vragen en meegaan (per speler)", () => {
  const ct = { type: "vraag", players: [0, 2] };
  assert.deepStrictEqual(pts(ct, [4, 0, 4, 5]), [2, -2, 2, -2]);     // 8 tricks: +2
  assert.deepStrictEqual(pts(ct, [4, 0, 3, 6]), [-6, 6, -6, 6]);     // 7: -6
  assert.deepStrictEqual(pts(ct, [6, 0, 6, 1]), [6, -6, 6, -6]);     // 12: +6
  assert.deepStrictEqual(pts(ct, [7, 0, 6, 0]), [14, -14, 14, -14]); // 13: 7 -> 14
  assert.strictEqual(R.vraagPts(1), -18);
});
test("alleen gaan (delen door 3)", () => {
  const ct = { type: "alleen", players: [1] };
  assert.deepStrictEqual(pts(ct, [3, 5, 3, 2]), [-2, 6, -2, -2]);
  assert.deepStrictEqual(pts(ct, [3, 4, 3, 3]), [4, -12, 4, 4]);
  assert.strictEqual(R.alleenPts(1), -30); assert.strictEqual(R.alleenPts(12), 27); assert.strictEqual(R.alleenPts(13), 60);
});
test("abondance 9/10/11/12", () => {
  for (const [lv, v] of [[9, 15], [10, 18], [11, 24], [12, 27]]) {
    const ct = { type: "abondance", players: [3], level: lv };
    assert.deepStrictEqual(pts(ct, [0, 0, 13 - lv, lv]), [-v / 3, -v / 3, -v / 3, v]);
    assert.deepStrictEqual(pts(ct, [0, 0, 14 - lv, lv - 1]), [v / 3, v / 3, v / 3, -v]);
  }
});
test("troel / troela", () => {
  assert.deepStrictEqual([8, 9, 12, 13, 7, 1].map(R.troelPts), [4, 6, 12, 28, -6, -18]);
  assert.deepStrictEqual([9, 10, 12, 13, 8, 1].map(R.troelaPts), [4, 6, 10, 24, -6, -20]);
});
test("solo / solo slim", () => {
  assert.deepStrictEqual(pts({ type: "solo", players: [0] }, [13, 0, 0, 0]), [75, -25, -25, -25]);
  assert.deepStrictEqual(pts({ type: "soloslim", players: [0] }, [12, 1, 0, 0]), [-90, 30, 30, 30]);
});
test("miserie grid (base 21) and miserie op tafel (base 42)", () => {
  assert.deepStrictEqual(pts({ type: "miserie", players: [0] }, [0, 5, 4, 4]), [21, -7, -7, -7]);
  assert.deepStrictEqual(pts({ type: "miserie", players: [0] }, [1, 4, 4, 4]), [-21, 7, 7, 7]);
  assert.deepStrictEqual(pts({ type: "miserie", players: [0, 1] }, [0, 2, 5, 6]), [28, -28, 0, 0]);
  assert.deepStrictEqual(pts({ type: "miserie", players: [0, 1, 2] }, [0, 1, 0, 12]), [21, -35, 21, -7]);
  assert.deepStrictEqual(pts({ type: "miserie", players: [0, 1, 2] }, [0, 1, 1, 11]), [35, -21, -21, 7]);
  assert.deepStrictEqual(pts({ type: "openmiserie", players: [0, 1, 2] }, [0, 0, 1, 12]), [42, 42, -70, -14]);
  assert.deepStrictEqual(pts({ type: "openmiserie", players: [2] }, [4, 4, 0, 5]), [-14, -14, 42, -14]);
});

// ---------------------------------------------------------------- auction
test("bidding: vraag + mee, overbid by abondance in trump suit", () => {
  const g = new R.Game({ rand: R.rngFrom(3), dealer: 3 });
  g.deal({ cut: 10, pattern: "445" });
  if (g.troelSeat != null) return; // rare: this seed has troel
  g.bid(0, { type: "vraag" });
  g.bid(1, { type: "mee" });
  const ts = g.turnedSuit, other = (ts + 1) % 4;
  assert.ok(g.legalBids(2).some((b) => b.type === "abondance" && b.level === 9 && b.trump === other));
  g.bid(2, { type: "abondance", level: 9, trump: other });
  assert.ok(g.legalBids(3).some((b) => b.type === "abondance" && b.level === 9 && b.trump === ts), "abondance in troef beats equal level");
  assert.ok(!g.legalBids(3).some((b) => b.type === "abondance" && b.level === 9 && b.trump !== ts));
});
test("bidding: nobody joins -> asker may go alone", () => {
  const g = new R.Game({ rand: R.rngFrom(5), dealer: 3 });
  g.deal({ cut: 20, pattern: "454" });
  if (g.troelSeat != null) return;
  g.bid(0, { type: "vraag" }); g.bid(1, { type: "pas" }); g.bid(2, { type: "pas" }); g.bid(3, { type: "pas" });
  assert.strictEqual(g.askAlone, 0);
  g.bid(0, { type: "alleen" });
  assert.strictEqual(g.contract.type, "alleen"); assert.strictEqual(g.contract.target, 5); assert.strictEqual(g.contract.leader, 0);
});
test("bidding: several players can play miserie", () => {
  const g = new R.Game({ rand: R.rngFrom(8), dealer: 3 });
  g.deal({});
  if (g.troelSeat != null) return;
  g.bid(0, { type: "miserie" }); g.bid(1, { type: "miserie" }); g.bid(2, { type: "pas" }); g.bid(3, { type: "pas" });
  assert.deepStrictEqual(g.contract.players, [0, 1]);
});
test("dealing 4-4-5 / 4-5-4: 13 cards each, dealer gets the turned card", () => {
  for (const pattern of ["445", "454"]) {
    const g = new R.Game({ rand: R.rngFrom(9), dealer: 2 });
    g.deal({ cut: 17, pattern });
    assert.ok(g.dealt.every((h) => h.length === 13));
    assert.ok(g.dealt[2].includes(g.turned));
  }
});

// ---------------------------------------------------------------- double dummy vs brute force
test("double-dummy solver = brute-force minimax on 400 endings", () => {
  const rand = R.rngFrom(11), dds = new DDS(18);
  const brute = (hands, trump, leader, teamA, counted, trick) => {
    const p = (leader + trick.length) % 4;
    if (!trick.length && !hands[0].length) return 0;
    const led = trick.length ? R.suitOf(trick[0]) : -1;
    let moves = hands[p];
    if (led >= 0 && moves.some((c) => R.suitOf(c) === led)) moves = moves.filter((c) => R.suitOf(c) === led);
    const max = (teamA >> p) & 1;
    let best = max ? -1 : 99;
    for (const c of moves) {
      const nh = hands.map((h, q) => (q === p ? h.filter((x) => x !== c) : h)), nt = trick.concat([c]);
      let v;
      if (nt.length === 4) {
        const w = R.trickWinner(nt.map((card, i) => ({ card, seat: (leader + i) % 4 })), trump);
        v = ((counted >> w) & 1) + brute(nh, trump, w, teamA, counted, []);
      } else v = brute(nh, trump, leader, teamA, counted, nt);
      best = max ? Math.max(best, v) : Math.min(best, v);
    }
    return best;
  };
  for (let t = 0; t < 400; t++) {
    const n = 1 + Math.floor(rand() * 4), d = R.shuffleInPlace([...Array(52).keys()], rand);
    const hands = [0, 1, 2, 3].map((p) => d.slice(p * n, (p + 1) * n));
    const trump = Math.floor(rand() * 5), leader = Math.floor(rand() * 4), teamA = 1 + Math.floor(rand() * 14);
    const counted = rand() < 0.5 ? teamA : 15 & ~teamA;
    assert.strictEqual(dds.solve(hands, trump, leader, teamA, counted), brute(hands, trump, leader, teamA, counted, []), `case ${t}`);
  }
});

// ---------------------------------------------------------------- card tracking
test("tracking: after an unshuffled deal hidden cards are located far better than 1/3", () => {
  const rand = R.rngFrom(42);
  let acc = 0, n = 0;
  for (let gi = 0; gi < 10; gi++) {
    const g = new R.Game({ rand, dealer: 0 });
    g.deal();
    while (g.phase === "bid") { const lb = g.legalBids(g.turn); g.bid(g.turn, lb.find((b) => ["vraag", "mee", "alleen"].includes(b.type)) || lb[0]); }
    if (g.phase !== "play") continue;
    while (g.phase === "play") { const lc = g.legalCards(g.turn); g.play(g.turn, lc[Math.floor(rand() * lc.length)]); }
    g.deal();
    const B = new Belief({ track: g.handTrack, seat: 1, dealer: g.dealer, pattern: g.pattern });
    const known = new Map(); g.dealt[1].forEach((c) => known.set(c, 1)); known.set(g.turned, g.dealer);
    B.update(known);
    const m = B.marginals(rand, 60);
    for (let p = 0; p < 4; p++) for (const c of g.dealt[p]) if (!known.has(c)) { acc += m[c][p]; n++; }
  }
  const mean = acc / n;
  console.log(`      mean P(true owner) = ${mean.toFixed(3)}`);
  assert.ok(mean > 0.6, "tracking too weak: " + mean);
});

test("habits: learned cut/gather/pickup habits sharpen tracking", () => {
  const { Habits } = require("../engine/belief.js");
  const rand = R.rngFrom(7);
  const g = new R.Game({ rand, dealer: 0 }), H = new Habits();
  let seen = 0, uni = 0, hab = 0, n = 0;
  for (let h = 0; h < 24; h++) {
    g.deal();
    if (h >= 14 && g.handTrack.kind === "tricks") {
      const seat = (h % 3) + 1;
      const known = new Map(); g.dealt[seat].forEach((c) => known.set(c, seat)); known.set(g.turned, g.dealer);
      for (const [hb, isH] of [[null, false], [H, true]]) {
        const B = new Belief({ track: g.handTrack, seat, dealer: g.dealer, pattern: g.pattern, habits: hb });
        B.update(known);
        const m = B.marginals(rand, 50);
        let a = 0, k = 0;
        for (let p = 0; p < 4; p++) for (const c of g.dealt[p]) if (!known.has(c)) { a += m[c][p]; k++; }
        if (isH) hab += a / k; else uni += a / k;
      }
      n++;
    }
    while (g.phase === "bid") { const lb = g.legalBids(g.turn); g.bid(g.turn, lb.find((b) => ["vraag", "mee", "alleen"].includes(b.type)) || lb[0]); }
    while (g.phase === "play") { const lc = g.legalCards(g.turn); g.play(g.turn, lc[Math.floor(rand() * lc.length)]); }
    for (; seen < g.history.length; seen++) H.learnFrom(g.history[seen]);
  }
  console.log(`      P(true owner): uniform ${(uni / n).toFixed(3)}, learned habits ${(hab / n).toFixed(3)}`);
  assert.ok(hab / n > uni / n + 0.02, "habits did not help");
});

test("bidding inference: auction replay gives each player's options at his turn", () => {
  const { auctionActions } = require("../engine/ai.js");
  const g = new R.Game({ rand: R.rngFrom(3), dealer: 3 });
  g.deal({ cut: 10, pattern: "445" });
  if (g.troelSeat != null) return;
  g.bid(0, { type: "vraag" }); g.bid(1, { type: "mee" }); g.bid(2, { type: "pas" });
  const acts = auctionActions(g, 3);
  assert.strictEqual(acts.length, 3);
  assert.ok(acts[0].legal.some((b) => b.type === "vraag") && !acts[0].legal.some((b) => b.type === "mee"));
  assert.ok(acts[1].legal.some((b) => b.type === "mee"));
  assert.ok(acts[2].legal.every((b) => b.type !== "vraag" && b.type !== "mee"));
  acts.forEach((a) => assert.ok(a.legal.some((b) => JSON.stringify(b) === a.chosen)));
});

console.log(failures ? `\n${failures} test(s) failed` : "\nall tests passed");
process.exit(failures ? 1 : 0);
