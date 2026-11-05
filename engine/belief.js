// belief.js — what one seat can infer about the hidden hands.
//
// The cards are not shuffled after the first game (Reglement §1): the new dealer gathers the
// trick piles, the previous dealer cuts (>= 4 cards off, >= 4 left), the dealer deals 4-4-5 or
// 4-5-4. All tricks of the previous hand were public, so the deck is a sequence of known
// 4-card blocks whose internal order is unknown. A hypothesis = (order in which the 4 piles were
// gathered, cut position). For each hypothesis we compute the exact likelihood of everything this
// seat has seen (own hand, turned card, cards played so far, open misery hands) and sample full
// deals from the posterior. Voids are enforced by rejection.
"use strict";
const R = typeof require !== "undefined" ? require("./rules.js") : globalThis;
const { suitOf, dealMap } = R;

const LOGF = [0];
for (let i = 1; i <= 52; i++) LOGF[i] = LOGF[i - 1] + Math.log(i);
const PERMS4 = [];
(function perm(a, k) {
  if (k === a.length) { PERMS4.push(a.slice()); return; }
  for (let i = k; i < a.length; i++) { [a[k], a[i]] = [a[i], a[k]]; perm(a, k + 1); [a[k], a[i]] = [a[i], a[k]]; }
})([0, 1, 2, 3], 0);

class Belief {
  // track: game.handTrack (public), seat: who is reasoning, dealer, pattern, prevOwnHand for "hands"
  constructor({ track, seat, dealer, pattern, prevOwnHand, useTracking = true }) {
    this.seat = seat;
    this.map = dealMap(dealer, pattern);
    this.hyps = []; // {blocks: [{pos:[...], cards:[...]|null}], unknownBlock?}
    const kind = useTracking ? track.kind : "shuffled";
    if (kind === "tricks") {
      const piles = [[], [], [], []];
      for (const t of track.tricks) piles[t.winner].push(t.cards);
      for (const order of PERMS4) {
        const blocks = [];
        let pos = 0;
        for (const p of order) for (const cards of piles[p]) { blocks.push({ pos: [pos, pos + 1, pos + 2, pos + 3], cards }); pos += 4; }
        this.addCuts(blocks);
      }
    } else if (kind === "hands" && prevOwnHand) {
      // only my own previous hand is a known block; the other 39 cards fill the other 39 positions
      const mine = new Set(prevOwnHand);
      const rest = [...Array(52).keys()].filter((c) => !mine.has(c));
      for (let k = 0; k < 4; k++) {   // where my pile was in the gathering order
        const myPos = [], otherPos = [];
        for (let i = 0; i < 52; i++) ((i >= 13 * k && i < 13 * (k + 1)) ? myPos : otherPos).push(i);
        this.addCuts([{ pos: myPos, cards: prevOwnHand.slice() }, { pos: otherPos, cards: rest }]);
      }
    } else {
      this.hyps.push({ blocks: [{ seats: this.map.slice(), cards: [...Array(52).keys()] }], cut: 0, w: 1 });
      this.kind = "shuffled";
      return;
    }
    this.kind = kind;
  }

  addCuts(blocks) {
    for (let cut = 4; cut <= 48; cut++) {
      // after cutting, old position i ends up at (i - cut) mod 52
      const bl = blocks.map((b) => ({ cards: b.cards, seats: b.pos.map((i) => this.map[(i - cut + 52) % 52]) }));
      this.hyps.push({ blocks: bl, cut, w: 0 });
    }
  }

  // known: Map card -> seat (my hand, turned card, played cards, open misery hands)
  update(known) {
    this.known = known;
    let maxL = -Infinity;
    for (const h of this.hyps) {
      let L = 0;
      for (const b of h.blocks) {
        const slots = [0, 0, 0, 0], kn = [0, 0, 0, 0];
        for (const s of b.seats) slots[s]++;
        let u = b.cards.length;
        for (const c of b.cards) { const o = known.get(c); if (o != null) { kn[o]++; u--; } }
        // all of my own cards are known: my slots must be filled exactly by my known cards
        if (kn[this.seat] !== slots[this.seat]) { L = -Infinity; break; }
        let ok = true;
        for (let p = 0; p < 4; p++) {
          if (kn[p] > slots[p]) { ok = false; break; }
          L += LOGF[slots[p]] - LOGF[slots[p] - kn[p]];
        }
        if (!ok) { L = -Infinity; break; }
        L += LOGF[u] - LOGF[b.cards.length];
      }
      h.logL = L;
      if (L > maxL) maxL = L;
    }
    let z = 0;
    for (const h of this.hyps) { h.w = h.logL === -Infinity ? 0 : Math.exp(h.logL - maxL); z += h.w; }
    for (const h of this.hyps) h.w /= z || 1;
    this.alive = this.hyps.filter((h) => h.w > 0);
    this.alive.sort((a, b) => b.w - a.w);
  }

  // effective number of hypotheses (1 = the deck order is pinned down)
  effective() { let s = 0; for (const h of this.alive) s += h.w * h.w; return 1 / s; }

  // remaining: number of cards each seat still holds; voids[p]: bitmask of suits p showed out of
  sampleWorld(rand, remaining, voids) {
    for (let attempt = 0; attempt < 60; attempt++) {
      const h = this.pickHyp(rand);
      const w = this.fillFromHyp(h, rand, voids);
      if (w) return w;
    }
    return this.uniformWorld(rand, remaining, voids);
  }

  pickHyp(rand) {
    let x = rand();
    for (const h of this.alive) { x -= h.w; if (x <= 0) return h; }
    return this.alive[0];
  }

  fillFromHyp(h, rand, voids) {
    const hands = [[], [], [], []];
    for (const b of h.blocks) {
      const free = [0, 0, 0, 0];
      for (const s of b.seats) free[s]++;
      const unk = [];
      for (const c of b.cards) { const o = this.known.get(c); if (o != null) free[o]--; else unk.push(c); }
      if (!assignRandom(unk, free, voids, rand, hands)) return null;
    }
    return hands; // unknown cards per seat (= current hidden holdings)
  }

  // ignore the deck structure: any assignment consistent with counts and voids
  uniformWorld(rand, remaining, voids) {
    const unk = [];
    for (let c = 0; c < 52; c++) if (!this.known.has(c)) unk.push(c);
    const free = [0, 0, 0, 0];
    for (let p = 0; p < 4; p++) {
      let k = 0;
      for (const [, o] of this.known) if (o === p) k++;
      free[p] = 13 - k;
    }
    for (let attempt = 0; attempt < 200; attempt++) {
      const hands = [[], [], [], []];
      if (assignRandom(unk, free.slice(), voids, rand, hands)) return hands;
    }
    const hands = [[], [], [], []]; // give up on voids
    assignRandom(unk, free.slice(), [0, 0, 0, 0], rand, hands);
    return hands;
  }

  // probability that each unknown card is held by each seat (for display / diagnostics)
  marginals(rand, n = 200, voids = [0, 0, 0, 0]) {
    const m = Array.from({ length: 52 }, () => [0, 0, 0, 0]);
    for (let i = 0; i < n; i++) {
      const w = this.sampleWorld(rand, null, voids);
      w.forEach((cards, p) => cards.forEach((c) => (m[c][p] += 1 / n)));
    }
    for (const [c, o] of this.known) { m[c] = [0, 0, 0, 0]; m[c][o] = 1; }
    return m;
  }
}

// assign cards to seats with free[p] slots, respecting voids; most constrained cards first
function assignRandom(cards, free, voids, rand, hands) {
  const order = cards.slice();
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  order.sort((a, b) => cnt(a) - cnt(b));
  function cnt(c) { let k = 0; for (let p = 0; p < 4; p++) if (!((voids[p] >> suitOf(c)) & 1)) k++; return k; }
  const f = free.slice();
  const out = [];
  for (const c of order) {
    let tot = 0;
    for (let p = 0; p < 4; p++) if (f[p] > 0 && !((voids[p] >> suitOf(c)) & 1)) tot += f[p];
    if (!tot) return false;
    let x = rand() * tot, pick = -1;
    for (let p = 0; p < 4; p++) if (f[p] > 0 && !((voids[p] >> suitOf(c)) & 1)) { x -= f[p]; if (x < 0) { pick = p; break; } }
    if (pick < 0) for (let p = 3; p >= 0; p--) if (f[p] > 0 && !((voids[p] >> suitOf(c)) & 1)) { pick = p; break; }
    f[pick]--;
    out.push([c, pick]);
  }
  for (const [c, p] of out) hands[p].push(c);
  for (let p = 0; p < 4; p++) free[p] = f[p];
  return true;
}

if (typeof module !== "undefined") module.exports = { Belief, assignRandom, PERMS4 };
