// ai.js — the Wiezen bot.
//   card play : PIMC. Sample hidden hands from the Belief (card tracking), and for every candidate
//               card play the hand out (fast heuristic for the early tricks, exact double-dummy
//               once every hand has <= K cards), score the result with the IWWA point table,
//               and pick the card with the best average score.
//   bidding   : the same rollouts estimate, per contract, the distribution of tricks; combined
//               with the WiezenLM network prior this gives an expected score for every legal bid.
"use strict";
const _req = typeof require !== "undefined";
const R = _req ? require("./rules.js") : globalThis;
const { DDS } = _req ? require("./dds.js") : globalThis;
const { Belief, Habits } = _req ? require("./belief.js") : globalThis;
const { suitOf, rankOf, NT, scoreHand, vraagPts, alleenPts, ABON_PTS, trickWinner } = R;

const K_DD = 7; // exact double-dummy once every hand holds <= K_DD cards

// ------------------------------------------------------------------ helpers
function maskOf(players) { let m = 0; for (const p of players) m |= 1 << p; return m; }
function isMisery(ct) { return ct.type === "miserie" || ct.type === "openmiserie"; }

// teams for the double-dummy solver, from `me`'s point of view
function ddSetup(ct, me) {
  const dm = maskOf(ct.players);
  if (isMisery(ct)) {
    if (ct.players.includes(me)) return { teamA: 15 & ~(1 << me), counted: 1 << me };
    return { teamA: 15 & ~dm, counted: dm };
  }
  return { teamA: dm, counted: dm };
}

// my points if the counted side ends with `total` tricks
function utility(ct, me, total) {
  const tricks = [0, 0, 0, 0];
  if (isMisery(ct)) {
    if (ct.players.includes(me)) tricks[me] = total; // other misery players assumed clean
    else { let left = total; for (const p of ct.players) { if (left > 0) { tricks[p] = 1; left--; } } if (left > 0) tricks[ct.players[0]] += left; }
  } else tricks[ct.players[0]] = total;
  return scoreHand(ct, tricks)[me];
}

function legalOf(hand, trick, forced) {
  if (forced != null && hand.includes(forced)) return [forced];
  if (trick.length) {
    const led = suitOf(trick[0].card);
    const f = hand.filter((c) => suitOf(c) === led);
    if (f.length) return f;
  }
  return hand;
}

// ------------------------------------------------------------------ heuristic policy
// Used inside sampled (perfect-information) worlds during rollouts, so it may look at all hands.
function heurCard(st, p) {
  const ct = st.ct, trump = ct.trump, hand = st.hands[p], trick = st.trick;
  const legal = legalOf(hand, trick, st.forced);
  if (legal.length === 1) return legal[0];
  const othersHave = (c) => { for (let q = 0; q < 4; q++) if (q !== p && st.hands[q].some((x) => suitOf(x) === suitOf(c) && x > c)) return true; return false; };
  const low = (cs) => cs.reduce((a, b) => (rankOf(b) < rankOf(a) ? b : a));
  const high = (cs) => cs.reduce((a, b) => (rankOf(b) > rankOf(a) ? b : a));
  const winner = trick.length ? trickWinner(trick, trump) : -1;
  const winCard = trick.length ? trick.find((x) => x.seat === winner).card : -1;
  const beats = (c) => (suitOf(c) === suitOf(winCard) && c > winCard) || (suitOf(c) === trump && suitOf(winCard) !== trump);
  const later = []; for (let i = trick.length + 1; i < 4; i++) later.push((st.leader + i) % 4);

  if (isMisery(ct)) {
    const mis = ct.players, iAmMis = mis.includes(p);
    if (iAmMis) {
      if (!trick.length) return low(legal);
      const under = legal.filter((c) => !beats(c));
      if (under.length) return high(under);
      return later.length ? low(legal) : high(legal);
    }
    if (!trick.length) {
      const target = legal.filter((c) => mis.some((m) => st.hands[m].some((x) => suitOf(x) === suitOf(c))));
      return low(target.length ? target : legal);
    }
    if (mis.includes(winner)) { const under = legal.filter((c) => !beats(c)); if (under.length) return high(under); }
    return low(legal);
  }

  const decl = ct.players, mySide = decl.includes(p);
  const opp = [0, 1, 2, 3].filter((q) => decl.includes(q) !== mySide);
  const canRuff = (s) => trump !== NT && s !== trump && opp.some((q) => !st.hands[q].some((x) => suitOf(x) === s) && st.hands[q].some((x) => suitOf(x) === trump));
  if (!trick.length) {
    const masters = legal.filter((c) => !othersHave(c) && !canRuff(suitOf(c)));
    if (masters.length) return high(masters);
    if (mySide && trump !== NT) {
      const myT = legal.filter((c) => suitOf(c) === trump);
      const oppT = opp.some((q) => st.hands[q].some((x) => suitOf(x) === trump));
      if (myT.length >= 2 && oppT) return othersHave(high(myT)) ? low(myT) : high(myT);
    }
    const bySuit = [0, 1, 2, 3].map((s) => legal.filter((c) => suitOf(c) === s && s !== trump));
    const longest = bySuit.reduce((a, b) => (b.length > a.length ? b : a));
    return low(longest.length ? longest : legal);
  }
  const partnerWins = decl.includes(winner) === mySide;
  const safe = (c) => later.every((q) => decl.includes(q) === mySide || !st.hands[q].some((x) => {
    if (suitOf(x) === suitOf(c)) return x > c;
    return suitOf(x) === trump && suitOf(c) !== trump && !st.hands[q].some((y) => suitOf(y) === suitOf(trick[0].card));
  }));
  const following = suitOf(legal[0]) === suitOf(trick[0].card);
  if (partnerWins && (later.length === 0 || safe(winCard))) {
    const nonT = legal.filter((c) => suitOf(c) !== trump || following);
    return low(nonT.length ? nonT : legal);
  }
  const winners = legal.filter(beats);
  if (winners.length) {
    const sw = winners.filter(safe);
    if (sw.length) return low(sw);
    if (!later.length) return low(winners);
  }
  const nonT = legal.filter((c) => suitOf(c) !== trump);
  return low(nonT.length ? nonT : legal);
}

// ------------------------------------------------------------------ rollout
// st: {hands (arrays, all 4 known), trick [{seat,card}], leader, turn, total (counted tricks so far), ct, forced}
function rollout(st, setup, dds) {
  const ct = st.ct;
  let total = st.total;
  for (;;) {
    const maxCards = Math.max(...st.hands.map((h) => h.length));
    if (maxCards === 0 && !st.trick.length) return total;
    const mis = setup.teamA !== setup.counted;
    const lim = Math.min(mis ? K_DD + 1 : K_DD, st.ddCap ?? 99);
    if (maxCards <= lim && !(mis && total > 0)) {
      const leader = st.trick.length ? st.trick[0].seat : st.leader, tc = st.trick.map((x) => x.card);
      dds.maxNodes = 120000;
      try {
        // misery: only "clean or not" matters -> cheap null-window search
        const v = mis ? dds.solve(st.hands, ct.trump, leader, setup.teamA, setup.counted, tc, 0, 1)
                      : dds.solve(st.hands, ct.trump, leader, setup.teamA, setup.counted, tc);
        return total + v;
      } catch (e) {
        if (e.message !== "budget") throw e;
        st.ddCap = maxCards - 1; // too expensive here: play on heuristically for a while
      } finally { dds.maxNodes = 0; }
    }
    if (mis && total > 0 && maxCards <= K_DD + 1) return total; // misery already lost
    const p = st.turn;
    const c = heurCard(st, p);
    st.hands[p] = st.hands[p].filter((x) => x !== c);
    st.forced = null;
    st.trick.push({ seat: p, card: c });
    if (st.trick.length === 4) {
      const w = trickWinner(st.trick, ct.trump);
      if ((setup.counted >> w) & 1) total++;
      st.trick = []; st.leader = w; st.turn = w;
    } else st.turn = (p + 1) % 4;
  }
}

// ------------------------------------------------------------------ public-information helpers
function knownMap(game, seat) {
  const known = new Map();
  for (const c of game.hands[seat]) known.set(c, seat);
  for (const t of game.tricks) t.cards.forEach((c, i) => known.set(c, t.seats[i]));
  for (const x of game.trick) known.set(x.card, x.seat);
  for (const c of game.dealt[seat]) known.set(c, seat);
  known.set(game.turned, game.dealer);
  if (game.openCards) for (const p of game.openCards) for (const c of game.dealt[p]) known.set(c, p);
  return known;
}
function voidsOf(game) {
  const v = [0, 0, 0, 0];
  const all = game.tricks.map((t) => t.cards.map((c, i) => ({ card: c, seat: t.seats[i] }))).concat([game.trick]);
  for (const tr of all) {
    if (!tr.length) continue;
    const led = suitOf(tr[0].card);
    for (const x of tr) if (suitOf(x.card) !== led) v[x.seat] |= 1 << led;
  }
  return v;
}
function makeBelief(game, seat, useTracking, habits = null) {
  const tr = game.handTrack;
  return new Belief({
    track: tr, seat, dealer: game.dealer, pattern: game.pattern, useTracking, habits: useTracking ? habits : null,
    prevOwnHand: tr.kind === "hands" && tr.prevHands ? tr.prevHands[seat] : null,
  });
}
// sample the hidden holdings and return full current hands
function sampleHands(game, seat, belief, rand, voids) {
  const w = belief.sampleWorld(rand, null, voids);
  const hands = [0, 1, 2, 3].map((p) => (p === seat ? game.hands[p].slice() : w[p].slice()));
  // cards of the other players that are known but not yet played (turned card, open misery)
  const known = belief.known;
  for (const [c, o] of known) if (o !== seat && game.hands[o].includes(c) && !hands[o].includes(c)) hands[o].push(c);
  return hands;
}

// ------------------------------------------------------------------ the bot
class Bot {
  constructor(opts = {}) {
    this.dds = opts.dds || new DDS(19);
    this.rand = opts.rand || Math.random;
    this.worlds = opts.worlds || 24;
    this.bidWorlds = opts.bidWorlds || 12;
    this.timeMs = opts.timeMs ?? 1500;     // time budget per decision (ms); at least 6 worlds are always used
    this.useTracking = opts.useTracking !== false;
    this.learnHabits = opts.learnHabits !== false; // learn cut/gather habits from finished hands
    this.prior = opts.prior || null;       // WiezenLM: (hand, turned, relSeat) -> trick distributions
    this.priorWeight = opts.priorWeight ?? 8;
  }

  // learn the players' cutting / gathering habits from all finished hands of this game
  syncHabits(game) {
    if (!this.learnHabits || !this.useTracking) return;
    if (this._gid !== game.gameId) { this.habits = new Habits(); this._gid = game.gameId; this._seen = 0; }
    for (; this._seen < game.history.length; this._seen++) this.habits.learnFrom(game.history[this._seen]);
  }

  belief(game, seat) {
    this.syncHabits(game);
    const key = game.gameId + ":" + game.handNo + ":" + seat;
    if (!this._b || this._bk !== key) { this._b = makeBelief(game, seat, this.useTracking, this.learnHabits ? this.habits : null); this._bk = key; }
    this._b.update(knownMap(game, seat));
    return this._b;
  }

  // ---------------- card play
  choosePlay(game, seat) {
    const legal = game.legalCards(seat);
    if (legal.length === 1) return { card: legal[0], values: null };
    const ct = game.contract;
    // collapse equivalent cards (adjacent among all cards still out)
    const out = new Set();
    for (let p = 0; p < 4; p++) for (const c of game.hands[p]) out.add(c);
    game.trick.forEach((x) => out.add(x.card));
    const cands = legal.filter((c) => {
      for (let r = rankOf(c) + 1; r < 13; r++) {
        const d = suitOf(c) * 13 + r;
        if (!out.has(d)) continue;
        return !legal.includes(d); // equivalent to the next higher card I also hold
      }
      return true;
    });
    if (cands.length === 1) return { card: cands[0], values: null };
    const belief = this.belief(game, seat);
    const voids = voidsOf(game);
    const setup = ddSetup(ct, seat);
    const sum = new Array(cands.length).fill(0);
    const total = ct.players ? [...Array(4).keys()].filter((p) => (setup.counted >> p) & 1).reduce((a, p) => a + game.won[p], 0) : 0;
    const forced = game.tricks.length === 0 && game.trick.length === 0 ? ct.forcedLead : null;
    const t0 = Date.now();
    let n = 0;
    for (let w = 0; w < this.worlds; w++) {
      if (this.timeMs && w >= 6 && Date.now() - t0 > this.timeMs) break;
      const hands = sampleHands(game, seat, belief, this.rand, voids);
      cands.forEach((c, i) => {
        const st = {
          hands: hands.map((h) => h.slice()), trick: game.trick.slice(), leader: game.leader, ct, total, forced: null,
        };
        st.hands[seat] = st.hands[seat].filter((x) => x !== c);
        st.trick.push({ seat, card: c });
        st.turn = (seat + 1) % 4;
        if (st.trick.length === 4) {
          const wv = trickWinner(st.trick, ct.trump);
          if ((setup.counted >> wv) & 1) st.total++;
          st.trick = []; st.leader = wv; st.turn = wv;
        }
        sum[i] += utility(ct, seat, rollout(st, setup, this.dds));
      });
      n++;
    }
    let bi = 0;
    for (let i = 1; i < cands.length; i++) if (sum[i] > sum[bi] + 1e-9) bi = i;
    return { card: cands[bi], values: cands.map((c, i) => ({ card: c, ev: sum[i] / n })), worlds: n, forced };
  }

  // ---------------- bidding
  // trick distributions (14 bins) for the contracts this seat could play
  estimate(game, seat) {
    const belief = this.belief(game, seat);
    const voor = (game.dealer + 1) % 4, ts = game.turnedSuit;
    const hist = () => new Array(14).fill(0);
    const S = { alone: [hist(), hist(), hist(), hist()], nt: hist(), alleen: hist(), pair: {}, vraag: hist(), mis: 0, n: 0 };
    const others = [0, 1, 2, 3].filter((p) => p !== seat);
    others.forEach((j) => (S.pair[j] = hist()));
    const voids = [0, 0, 0, 0];
    const t0 = Date.now();
    for (let w = 0; w < this.bidWorlds; w++) {
      if (this.timeMs && w >= 4 && Date.now() - t0 > this.timeMs * 1.5) break;
      const hands = sampleHands(game, seat, belief, this.rand, voids);
      const run = (ct, leader) => {
        const setup = ddSetup(ct, seat);
        return rollout({ hands: hands.map((h) => h.slice()), trick: [], leader, turn: leader, total: 0, ct, forced: null }, setup, this.dds);
      };
      for (let t = 0; t < 4; t++) S.alone[t][run({ type: "abondance", players: [seat], trump: t, level: 9 }, seat)]++;
      S.nt[run({ type: "solo", players: [seat], trump: NT }, seat)]++;
      S.alleen[run({ type: "alleen", players: [seat], trump: ts }, voor)]++;
      S.mis += run({ type: "miserie", players: [seat], trump: NT }, voor) === 0 ? 1 : 0;
      let jBest = others[0], best = -1;
      for (const j of others) {
        const k = run({ type: "vraag", players: [seat, j], trump: ts }, voor);
        S.pair[j][k]++;
        const strength = hands[j].filter((c) => suitOf(c) === ts).length * 2 + hands[j].filter((c) => rankOf(c) >= 9).length;
        if (strength > best) { best = strength; jBest = j; S._k = k; }
      }
      S.vraag[S._k]++;
      S.n++;
    }
    // blend with the network prior; trust samples more when tracking pins the deal down
    const D = (h) => h.map((x) => x / S.n);
    const out = { alone: S.alone.map(D), nt: D(S.nt), alleen: D(S.alleen), vraag: D(S.vraag), pair: {}, mis: S.mis / S.n, n: S.n };
    others.forEach((j) => (out.pair[j] = D(S.pair[j])));
    if (this.prior) {
      const rel = (seat - game.dealer + 4) % 4;
      const P = this.prior(game.dealt[seat], game.turned, rel);
      const eff = belief.kind === "shuffled" ? 999 : belief.effective();
      const wp = belief.kind === "shuffled" ? this.priorWeight : Math.min(this.priorWeight, eff / 4);
      const mix = (a, b) => a.map((x, i) => (S.n * x + wp * b[i]) / (S.n + wp));
      for (let t = 0; t < 4; t++) out.alone[t] = mix(out.alone[t], P.alone[t]);
      out.nt = mix(out.nt, P.nt);
      out.alleen = mix(out.alleen, P.alleen);
      others.forEach((j) => (out.pair[j] = mix(out.pair[j], P.pair[(j - seat + 4) % 4 - 1])));
      out.vraag = mix(out.vraag, P.pair.reduce((a, b) => (expect(b) > expect(a) ? b : a)));
      out.mis = (S.n * out.mis + wp * P.mis) / (S.n + wp);
      out.prior = P;
    }
    return out;
  }

  bidValues(game, seat, E) {
    const ts = game.turnedSuit;
    const ev = (dist, f) => dist.reduce((a, p, k) => a + p * f(k), 0);
    const pMisOpen = E.mis, pMis = 1 - (1 - E.mis) * 0.75; // opponents cannot see a closed misery hand
    const val = (b) => {
      switch (b.type) {
        case "pas": return 0;
        case "vraag": return ev(E.vraag, vraagPts);
        case "mee": return ev(E.pair[game.holders[0]], vraagPts);
        case "alleen": return ev(E.alleen, alleenPts);
        case "abondance": return ev(E.alone[b.trump], (k) => (k >= b.level ? ABON_PTS[b.level] : -ABON_PTS[b.level]));
        case "miserie": return 21 * (2 * pMis - 1);
        case "openmiserie": return 42 * (2 * pMisOpen - 1);
        case "solo": return 75 * (2 * (b.trump === NT ? E.nt : E.alone[b.trump])[13] - 1);
        case "soloslim": return 90 * (2 * E.alone[ts][13] - 1);
      }
      return -99;
    };
    return val;
  }

  chooseBid(game, seat) {
    const legal = game.legalBids(seat);
    if (legal.length === 1) return { bid: legal[0] };
    const E = this.estimate(game, seat);
    const val = this.bidValues(game, seat, E);
    let best = legal[0], bv = 0.5; // small margin: only bid when clearly positive
    const scored = legal.map((b) => ({ bid: b, ev: val(b) }));
    for (const s of scored) if (s.bid.type !== "pas" && s.ev > bv) { best = s.bid; bv = s.ev; }
    return { bid: best, values: scored, estimate: E };
  }
}

function expect(d) { return d.reduce((a, p, k) => a + p * k, 0); }

if (typeof module !== "undefined") module.exports = { Bot, heurCard, rollout, ddSetup, utility, knownMap, voidsOf, makeBelief, sampleHands, K_DD, legalOf };
