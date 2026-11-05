// dds.js — double-dummy solver for 4-player whist (all hands open, perfect play).
// Teams are arbitrary: `teamA` (bitmask of seats) maximizes the number of tricks won by the seats in
// `counted` (bitmask); everyone else minimizes it. That covers 2v2 (vraag/meegaan, troel), 1v3
// (alleen, abondance, solo) and misery (teamA = opponents, counted = misery player).
// Alpha-beta with a transposition table at trick boundaries and equivalent-card pruning.
"use strict";

function popcnt(x) { x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333); return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24; }

class DDS {
  constructor(ttBits = 20) {
    this.size = 1 << ttBits;
    this.mask = this.size - 1;
    this.tk1 = new Int32Array(this.size);
    this.tk2 = new Int32Array(this.size);
    this.tgen = new Int32Array(this.size);
    this.tlb = new Int8Array(this.size);
    this.tub = new Int8Array(this.size);
    this.tbm = new Int8Array(this.size);  // best lead: suit*16 + position from the top of the suit
    this.gen = 0;
    let s = 0x12345678 | 0;
    const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return s | 0; };
    this.z1 = new Int32Array(4 * 52); this.z2 = new Int32Array(4 * 52);
    for (let i = 0; i < 4 * 52; i++) { this.z1[i] = rnd(); this.z2[i] = rnd(); }
    this.zl1 = new Int32Array(4); this.zl2 = new Int32Array(4);
    for (let i = 0; i < 4; i++) { this.zl1[i] = rnd(); this.zl2[i] = rnd(); }
    this.h = new Int32Array(16);      // h[p*4+s] = 13-bit rank mask
    this.tc = new Int32Array(4);      // current trick cards
    this.mv = new Int32Array(53 * 13);  // move buffers per depth
    this.ms = new Int32Array(13);
    this.nodes = 0;
    this.maxNodes = 0; // 0 = unlimited; otherwise solve() throws "budget" when exceeded
  }

  // hands: 4 arrays of card ids (s*13+r). trick: cards already in the current trick, in play order,
  // the first played by `leader`. Returns tricks for `counted` from the current trick (inclusive) onward.
  solve(hands, trump, leader, teamA, counted, trick = [], alpha = -1, beta = 99) {
    const h = this.h;
    h.fill(0);
    this.k1 = 0; this.k2 = 0;
    let ncards = 0;
    for (let p = 0; p < 4; p++)
      for (const c of hands[p]) {
        h[p * 4 + ((c / 13) | 0)] |= 1 << c % 13;
        this.k1 ^= this.z1[p * 52 + c]; this.k2 ^= this.z2[p * 52 + c];
        ncards++;
      }
    this.trump = trump; this.teamA = teamA; this.counted = counted;
    this.normal = counted === teamA;  // both sides want to win tricks (not misery)
    this.leader = leader; this.nIn = 0;
    this.gen++;
    this.depth = 0;
    this.limit = this.maxNodes ? this.nodes + this.maxNodes : Infinity;
    for (const c of trick) {
      const s = (c / 13) | 0, r = c % 13, p = (leader + this.nIn) % 4;
      this.tc[this.nIn] = c;
      if (this.nIn === 0 || (s === this.winS && r > this.winR) || (s === trump && this.winS !== trump)) {
        this.winS = s; this.winR = r; this.winP = p;
      }
      this.nIn++;
    }
    this.remTricks = (ncards + this.nIn) / 4;
    if (alpha > -1 || beta < 99) return this.ab(alpha, beta);  // explicit window requested
    return this.mtdf(this.guess != null ? Math.min(this.guess, this.remTricks) : (this.remTricks >> 1));
  }

  // exact value by a sequence of null-window searches (TT persists across them)
  mtdf(g) {
    let lo = 0, hi = this.remTricks;
    while (lo < hi) {
      const b = g === lo ? g + 1 : g;
      const v = this.ab(b - 1, b);
      if (v < b) hi = v; else lo = v;
      g = v;
    }
    return lo;
  }

  ab(alpha, beta) {
    if (++this.nodes > this.limit) throw new Error("budget");
    const h = this.h, nIn = this.nIn;
    let lb = 0, ub = this.remTricks, idx = -1, key1 = 0, key2 = 0, hint = -1;
    if (nIn === 0) {
      if (ub === 0) return 0;
      // key on relative ranks: positions that differ only by already-played cards share an entry
      key1 = this.leader + 1; key2 = (this.leader + 7) * 0x9e3779b1;
      for (let s = 0; s < 4; s++) {
        const all = h[s] | h[4 + s] | h[8 + s] | h[12 + s];
        let c0 = 0, c1 = 0, c2 = 0, c3 = 0, bit = 1, rest = all;
        while (rest) {
          const low = rest & -rest;
          if (h[s] & low) c0 |= bit; else if (h[4 + s] & low) c1 |= bit; else if (h[8 + s] & low) c2 |= bit; else c3 |= bit;
          bit <<= 1; rest ^= low;
        }
        key1 = Math.imul(key1 ^ c0, 0x01000193); key1 = Math.imul(key1 ^ c1, 0x01000193);
        key1 = Math.imul(key1 ^ c2, 0x01000193); key1 = Math.imul(key1 ^ (c3 | (bit << 13)), 0x01000193);
        key2 = Math.imul(key2 ^ (c0 | (c1 << 13)), 0x85ebca6b); key2 ^= key2 >>> 13;
        key2 = Math.imul(key2 ^ (c2 | (c3 << 13)), 0xc2b2ae35); key2 ^= key2 >>> 16;
      }
      idx = key1 & this.mask;
      if (this.tgen[idx] === this.gen && this.tk1[idx] === key1 && this.tk2[idx] === key2) {
        lb = this.tlb[idx]; ub = this.tub[idx]; hint = this.tbm[idx];
      }
      if (this.normal) {  // quick tricks: top winners the leader can cash right now
        const L = this.leader, side = (this.teamA >> L) & 1, tr = this.trump;
        let oppTrumps = false;
        if (tr < 4) for (let q = 0; q < 4; q++) if (((this.teamA >> q) & 1) !== side && h[q * 4 + tr]) oppTrumps = true;
        let qt = 0;
        for (let s = 0; s < 4; s++) {
          if (oppTrumps && s !== tr) continue;
          const m = h[L * 4 + s];
          if (!m) continue;
          let rest = h[s] | h[4 + s] | h[8 + s] | h[12 + s];
          while (rest) {
            const top = 1 << (31 - Math.clz32(rest));
            if (!(m & top)) break;
            qt++; rest ^= top;
          }
        }
        if (qt > ub) qt = ub;
        if (side) { if (qt > lb) lb = qt; }
        else if (ub - qt < ub) { const u = this.remTricks - qt; if (u < ub) ub = u; }
      }
      // also applies to the trivial bounds [0, remaining]: an empty window must not be searched
      if (lb >= beta) return lb;
      if (ub <= alpha) return ub;
      if (lb === ub) return lb;
      if (alpha < lb) alpha = lb;
      if (beta > ub) beta = ub;
    }
    const p = (this.leader + nIn) % 4;
    const isMax = (this.teamA >> p) & 1;
    const trump = this.trump;
    // ---- move generation (representatives of equivalent runs), ordered
    const base = this.depth * 13;
    let n = 0;
    let led = -1, follow = false;
    if (nIn > 0) { led = (this.tc[0] / 13) | 0; follow = h[p * 4 + led] !== 0; }
    const partnerWinning = nIn > 0 && ((this.teamA >> this.winP) & 1) === isMax;
    for (let s = 0; s < 4; s++) {
      if (follow && s !== led) continue;
      const m = h[p * 4 + s];
      if (!m) continue;
      let all = h[s] | h[4 + s] | h[8 + s] | h[12 + s];
      for (let i = 0; i < nIn; i++) { const c = this.tc[i]; if (((c / 13) | 0) === s) all |= 1 << c % 13; }
      for (let r = 12; r >= 0; r--) {
        if (!((m >> r) & 1)) continue;
        const higher = all >>> (r + 1);
        if (higher) {
          const nr = r + 1 + (31 - Math.clz32(higher & -higher));
          if ((m >> nr) & 1) continue;  // equivalent to the next higher card in this hand
        }
        let sc;
        if (nIn === 0) {
          sc = r + (s === trump ? 1 : 0);
          if (hint >= 0 && (hint >> 4) === s && (hint & 15) === popcnt(all >>> (r + 1))) sc += 1000;
        }
        else {
          const wins = (s === this.winS && r > this.winR) || (s === trump && this.winS !== trump);
          if (partnerWinning) sc = s === trump && !follow ? -100 - r : -r;
          else if (wins) sc = 100 - r;
          else sc = s === trump && !follow ? -60 - r : -r;
        }
        // insertion sort by score desc
        let j = n++;
        while (j > 0 && this.ms[j - 1] < sc) { this.ms[j] = this.ms[j - 1]; this.mv[base + j] = this.mv[base + j - 1]; j--; }
        this.ms[j] = sc; this.mv[base + j] = s * 13 + r;
      }
    }
    // ---- search
    const a0 = alpha, b0 = beta;
    let best = isMax ? -1 : 99, bestCard = -1;
    const sWinS = this.winS, sWinR = this.winR, sWinP = this.winP, sLeader = this.leader;
    for (let i = 0; i < n; i++) {
      const c = this.mv[base + i], s = (c / 13) | 0, r = c % 13;
      h[p * 4 + s] &= ~(1 << r);
      this.k1 ^= this.z1[p * 52 + c]; this.k2 ^= this.z2[p * 52 + c];
      this.tc[nIn] = c;
      if (nIn === 0 || (s === this.winS && r > this.winR) || (s === trump && this.winS !== trump)) {
        this.winS = s; this.winR = r; this.winP = p;
      }
      this.depth++;
      let v;
      if (nIn === 3) {
        const w = this.winP, inc = (this.counted >> w) & 1;
        const t0 = this.tc[0], t1 = this.tc[1], t2 = this.tc[2];  // next trick overwrites tc
        this.leader = w; this.nIn = 0; this.remTricks--;
        v = inc + this.ab(alpha - inc, beta - inc);
        this.remTricks++; this.nIn = 3; this.leader = sLeader;
        this.tc[0] = t0; this.tc[1] = t1; this.tc[2] = t2;
      } else {
        this.nIn = nIn + 1;
        v = this.ab(alpha, beta);
        this.nIn = nIn;
      }
      this.depth--;
      this.winS = sWinS; this.winR = sWinR; this.winP = sWinP;
      h[p * 4 + s] |= 1 << r;
      this.k1 ^= this.z1[p * 52 + c]; this.k2 ^= this.z2[p * 52 + c];
      if (isMax ? v > best : v < best) { best = v; bestCard = c; }
      if (isMax) { if (best > alpha) alpha = best; }
      else if (best < beta) beta = best;
      if (alpha >= beta) break;
    }
    if (idx >= 0) {
      let nlb = lb, nub = ub;
      if (best <= a0) nub = Math.min(nub, best);
      else if (best >= b0) nlb = Math.max(nlb, best);
      else { nlb = best; nub = best; }
      this.tk1[idx] = key1; this.tk2[idx] = key2; this.tgen[idx] = this.gen;
      this.tlb[idx] = nlb; this.tub[idx] = nub;
      if (bestCard >= 0) {
        const bs = (bestCard / 13) | 0, all = h[bs] | h[4 + bs] | h[8 + bs] | h[12 + bs];
        this.tbm[idx] = bs * 16 + popcnt(all >>> (bestCard % 13 + 1));
      } else this.tbm[idx] = -1;
    }
    return best;
  }
}

if (typeof module !== "undefined") module.exports = { DDS };
