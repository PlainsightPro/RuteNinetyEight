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
  constructor({ track, seat, dealer, pattern, prevOwnHand, useTracking = true, habits = null }) {
    this.seat = seat;
    this.habits = habits;
    this.gatherer = dealer;           // the dealer of this hand gathered the previous hand's piles
    this.cutter = (dealer + 3) % 4;   // the previous dealer cut
    this.map = dealMap(dealer, pattern);
    this.hyps = []; // {blocks: [{pos:[...], cards:[...]|null}], unknownBlock?}
    const kind = useTracking ? track.kind : "shuffled";
    if (kind === "tricks") {
      const piles = [[], [], [], []];
      for (const t of track.tricks) piles[t.winner].push({ cards: t.cards, fixed: fixedOrders(t), winner: t.winner });
      this.mix = habits ? [0, 1, 2, 3].map((w) => habits.pickMix(w)) : null;
      for (const order of PERMS4) {
        const blocks = [];
        let pos = 0;
        for (const p of order) for (const t of piles[p]) { blocks.push({ pos: [pos, pos + 1, pos + 2, pos + 3], cards: t.cards, fixed: t.fixed, winner: t.winner }); pos += 4; }
        this.addCuts(blocks, order);
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

  addCuts(blocks, order = null) {
    for (let cut = 4; cut <= 48; cut++) {
      // after cutting, old position i ends up at (i - cut) mod 52
      const bl = blocks.map((b) => ({ cards: b.cards, fixed: b.fixed, winner: b.winner, seats: b.pos.map((i) => this.map[(i - cut + 52) % 52]) }));
      const lp = this.habits ? this.habits.logPrior(order, cut, this.gatherer, this.cutter) : 0;
      this.hyps.push({ blocks: bl, cut, order, lp, w: 0 });
    }
  }

  // Likelihood of what this seat knows, for one block of one hypothesis. A block's cards are either
  // stacked in one of the fixed orders (play order, reverse, by seat; see fixedOrders) or sloppily.
  // Returns [P(observation | sloppy), and per fixed order 1 if consistent else 0].
  blockParts(b, known) {
    const slots = [0, 0, 0, 0], kn = [0, 0, 0, 0];
    for (const s of b.seats) slots[s]++;
    let u = b.cards.length, pu = 0;
    for (const c of b.cards) { const o = known.get(c); if (o != null) { kn[o]++; u--; } }
    // sloppy: all my own cards are known, so my slots must be filled exactly by my known cards
    if (kn[this.seat] === slots[this.seat]) {
      let L = 0, ok = true;
      for (let p = 0; p < 4; p++) { if (kn[p] > slots[p]) { ok = false; break; } L += LOGF[slots[p]] - LOGF[slots[p] - kn[p]]; }
      if (ok) pu = Math.exp(L + LOGF[u] - LOGF[b.cards.length]);
    }
    if (!this.mix || !b.fixed) return [pu, null];
    const fx = b.fixed.map((ord) => {
      for (let i = 0; i < 4; i++) {
        const o = known.get(ord[i]), s = b.seats[i];
        if (o != null ? o !== s : s === this.seat) return 0;
      }
      return 1;
    });
    return [pu, fx];
  }
  blockLik(b, known) {
    const [pu, fx] = this.blockParts(b, known);
    if (!fx) return pu;
    const m = this.mix[b.winner];
    let L = m[m.length - 1] * pu;
    for (let i = 0; i < fx.length; i++) if (fx[i]) L += m[i];
    return L;
  }

  // known: Map card -> seat (my hand, turned card, played cards, open misery hands)
  update(known) {
    this.known = known;
    let maxL = -Infinity;
    for (const h of this.hyps) {
      let L = 0;
      for (const b of h.blocks) {
        const l = this.blockLik(b, known);
        if (l <= 0) { L = -Infinity; break; }
        L += Math.log(l);
      }
      h.logL = L + (h.lp || 0);
      if (h.logL > maxL) maxL = h.logL;
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
      if (this.mix && b.fixed) {  // choose how this block was stacked, given what is known
        const [pu, fx] = this.blockParts(b, this.known), m = this.mix[b.winner];
        let tot = m[m.length - 1] * pu;
        for (let i = 0; i < fx.length; i++) tot += fx[i] * m[i];
        let x = rand() * tot, mode = -1;
        for (let i = 0; i < fx.length; i++) { x -= fx[i] * m[i]; if (x < 0) { mode = i; break; } }
        if (mode >= 0) {
          const ord = b.fixed[mode];
          for (let i = 0; i < 4; i++) {
            const c = ord[i], s = b.seats[i];
            if (this.known.has(c)) continue;
            if ((voids[s] >> suitOf(c)) & 1) return null;
            hands[s].push(c);
          }
          continue;
        }
      }
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

// The plausible tidy ways to stack a trick: in play order, reversed, or by seat relative to the
// winner (24 orders, e.g. "own card first, then clockwise"). Index 0 = play order, 1 = reverse.
function fixedOrders(t) {
  const out = [t.cards.slice(), t.cards.slice().reverse()];
  if (t.seats) for (const perm of PERMS4) out.push(perm.map((off) => t.cards[t.seats.indexOf((t.winner + off) % 4)]));
  return out;
}
const N_FIXED = 26;

// ------------------------------------------------------------------ learned habits
// After a hand all 52 cards have been played, so everyone can reconstruct how the deck for that hand
// was gathered and cut. Habits keeps, per player, a distribution over cut depth and over the order in
// which they gather the piles (start seat + clockwise, or anything else), and turns it into a prior.
class Habits {
  constructor() {
    // weak generic prior: people tend to cut somewhere near the middle
    // (worth about 3 observations, so a player's own habit takes over after a few hands)
    this.cutPrior = (() => {
      const a = new Float64Array(53); let t = 0;
      for (let c = 4; c <= 48; c++) { a[c] = Math.exp(-0.5 * ((c - 26) / 11) ** 2); t += a[c]; }
      for (let c = 4; c <= 48; c++) a[c] = (3 * a[c]) / t + 0.01;
      return a;
    })();
    this.gatherPrior = [0.4, 0.4, 0.4, 0.4, 1.6]; // rotations k = 0..3, other
    this.cut = [0, 1, 2, 3].map(() => this.cutPrior.slice());
    this.gather = [0, 1, 2, 3].map(() => this.gatherPrior.slice());
    this.hands = []; // per finished hand: equivalent explanations {cut, rot, w}
    // how each player stacks the tricks he wins: 26 tidy orders + sloppy (pseudo-counts)
    this.pickPrior = (() => { const a = new Array(N_FIXED + 1).fill(0.6 / 24); a[0] = 0.8; a[1] = 0.2; a[N_FIXED] = 2.4; return a; })();
    this.pick = [0, 1, 2, 3].map(() => this.pickPrior.slice());
    this.evidence = [[], [], [], []]; // per trick winner: {w, fx, pu} from finished hands
    this.learned = 0;
  }
  // EM over all finished hands: which cut depth and gathering start does each player prefer?
  // The likelihood has local optima (see learnFrom), so restart from several random habit guesses
  // and keep the fit that explains the hands best.
  refitDeck(iters = 20, restarts = 10) {
    if (!this.hands.length) return;
    let s = 12345 + this.hands.length;
    const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    let best = null;
    for (let r = 0; r < restarts; r++) {
      let cut = [0, 1, 2, 3].map(() => this.cutPrior.slice()), gat = [0, 1, 2, 3].map(() => this.gatherPrior.slice());
      if (r > 0) for (let p = 0; p < 4; p++) {
        const mu = 8 + rnd() * 36;
        for (let c = 4; c <= 48; c++) cut[p][c] += 3 * Math.exp(-0.5 * ((c - mu) / 4) ** 2);
        gat[p][Math.floor(rnd() * 4)] += 2;
      }
      let ll = 0;
      for (let it = 0; it <= iters; it++) {
        const nc = [0, 1, 2, 3].map(() => this.cutPrior.slice()), ng = [0, 1, 2, 3].map(() => this.gatherPrior.slice());
        const ct = cut.map((a) => a.reduce((x, y) => x + y, 0)), gt = gat.map((a) => a.reduce((x, y) => x + y, 0));
        ll = 0;
        for (const hd of this.hands) {
          const cm = cut[hd.cutter], gm = gat[hd.gatherer];
          const post = hd.alts.map((a) => a.w * (cm[a.cut] / ct[hd.cutter]) * ((a.rot === 4 ? 0 : gm[a.rot]) + gm[4] / 24) / gt[hd.gatherer]);
          const T = post.reduce((x, y) => x + y, 0);
          if (!(T > 0)) continue;
          ll += Math.log(T);
          hd.alts.forEach((a, i) => { nc[hd.cutter][a.cut] += (2 * post[i]) / T; ng[hd.gatherer][a.rot] += post[i] / T; });
        }
        if (it < iters) { cut = nc; gat = ng; }
      }
      if (!best || ll > best.ll) best = { ll, cut, gat };
    }
    this.cut = best.cut; this.gather = best.gat;
  }

  // EM over all stored tricks: which tidy order (or sloppiness) does each player use?
  refitPickup(iters = 12) {
    for (let p = 0; p < 4; p++) {
      const ev = this.evidence[p];
      let mix = this.pickMix(p);
      for (let it = 0; it < iters; it++) {
        const cnt = this.pickPrior.slice();
        for (const e of ev) {
          let T = mix[N_FIXED] * e.pu;
          for (let i = 0; i < N_FIXED; i++) if (e.fx[i]) T += mix[i];
          if (T <= 0) continue;
          for (let i = 0; i < N_FIXED; i++) if (e.fx[i]) cnt[i] += (e.w * mix[i]) / T;
          cnt[N_FIXED] += (e.w * mix[N_FIXED] * e.pu) / T;
        }
        this.pick[p] = cnt;
        mix = this.pickMix(p);
      }
    }
  }
  pickMix(w) { const a = this.pick[w], T = a.reduce((x, y) => x + y, 0); return a.map((x) => x / T); }
  static rotation(order, gatherer) {
    const k = (order[0] - gatherer + 4) % 4;
    return order.every((p, i) => p === (gatherer + k + i) % 4) ? k : 4;
  }
  logPrior(order, cut, gatherer, cutter) {
    const a = this.cut[cutter];
    let tot = 0;
    for (let c = 4; c <= 48; c++) tot += a[c];
    let lp = Math.log(a[cut] / tot);
    if (order) {
      const g = this.gather[gatherer], T = g[0] + g[1] + g[2] + g[3] + g[4], k = Habits.rotation(order, gatherer);
      lp += Math.log((k < 4 ? g[k] / T : 0) + g[4] / T / 24);
    }
    return lp;
  }
  // entry: a finished hand from game.history (dealt hands are public after the hand)
  learnFrom(entry) {
    if (!entry.track || entry.track.kind !== "tricks" || !entry.pattern) return;
    const B = new Belief({ track: entry.track, seat: 0, dealer: entry.dealer, pattern: entry.pattern, habits: this });
    const known = new Map();
    entry.dealt.forEach((h, p) => h.forEach((c) => known.set(c, p)));
    B.update(known);
    const gatherer = entry.dealer, cutter = (entry.dealer + 3) % 4;
    // Several (gather order, cut) pairs produce exactly the same deck (start one pile later, cut one
    // pile less), so keep all explanations and resolve them jointly with EM over all hands.
    const alts = [];
    for (const h of B.alive) if (h.w > 1e-6) alts.push({ cut: h.cut, rot: Habits.rotation(h.order, gatherer), w: h.w });
    this.hands.push({ gatherer, cutter, alts });
    const top = B.alive[0];
    for (const h of top ? [top] : []) {  // equivalent explanations give the same deck: one is enough
      h.w = 1;
      for (const b of h.blocks) {  // evidence on how each trick was stacked
        if (!b.fixed) continue;
        const [pu, fx] = B.blockParts(b, known);
        if (fx.every((f) => f) && pu >= 0.999) continue; // whole trick went to one player: no information
        this.evidence[b.winner].push({ w: h.w, fx, pu });
      }
    }
    this.refitPickup();
    this.refitDeck();
    this.learned++;
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

if (typeof module !== "undefined") module.exports = { Belief, Habits, assignRandom, PERMS4 };
