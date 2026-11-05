// rules.js — Wiezen according to the IWWA "Specifiek Reglement Wiezen" (versie 4, 25-08-2026)
// and the IWWA "Puntentabel Wiezen/Whist" (Federatie versie 1, 01-10-2025). See docs/.
//
// Seats 0..3 clockwise. Card id = suit*13 + rank, suits 0♣ 1♦ 2♥ 3♠, rank 0 = 2 ... 12 = ace.
// Choices where the regulations are silent are marked "ASSUMPTION" (also listed in README).
"use strict";

const SUIT_SYM = ["♣", "♦", "♥", "♠"];
const SUIT_NAME = ["klaveren", "ruiten", "harten", "schoppen"];
const RANK_STR = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "B", "V", "H", "A"]; // boer, vrouw, heer, aas
const suitOf = (c) => (c / 13) | 0;
const rankOf = (c) => c % 13;
const cardStr = (c) => RANK_STR[rankOf(c)] + SUIT_SYM[suitOf(c)];
const NT = 4; // no trump (solo "zonder troef", miserie)

// ------------------------------------------------------------------ scoring (Puntentabel)
const ABON_PTS = { 9: 15, 10: 18, 11: 24, 12: 27 };
// "Vragen en meegaan" per player. Table rows 1..13; 0 tricks extrapolated (ASSUMPTION).
function vraagPts(t) { return t === 13 ? 14 : t >= 8 ? 2 + (t - 8) : -6 - 2 * (7 - t); }
// "Alleen gaan" (5 tricks): declarer gets this, divided over the 3 opponents.
function alleenPts(t) { return t === 13 ? 60 : t >= 5 ? 6 + 3 * (t - 5) : -12 - 6 * (4 - t); }
// Troel (3 aces, 8 tricks) and Troela (4 aces, 9 tricks), per player.
function troelPts(t) { return t === 13 ? 28 : t >= 8 ? 4 + 2 * (t - 8) : -6 - 2 * (7 - t); }
function troelaPts(t) { return t === 13 ? 24 : t >= 9 ? 4 + 2 * (t - 9) : -6 - 2 * (8 - t); }
// Miserie grid (base 21; miserie op tafel = base 42 = everything doubled).
// Returns points for each misery player (by success) and for each non-misery player.
function miseryPts(n, s) {
  // [points per successful misery player, per failed misery player, per other player]
  const T = {
    1: { 0: [0, -21, 7], 1: [21, 0, -7] },
    2: { 0: [0, -14, 14], 1: [28, -28, 0], 2: [14, 0, -14] },
    3: { 0: [0, -7, 21], 1: [35, -21, 7], 2: [21, -35, -7], 3: [7, 0, -21] },
  };
  if (T[n]) return T[n][s];
  return [7 * (4 - s), -7 * s, 0]; // 4 misery players: not in the table (ASSUMPTION, zero-sum)
}

// points per seat for a finished contract, given tricks won by each seat
function scoreHand(ct, tricks) {
  const pts = [0, 0, 0, 0];
  const decl = ct.players;
  const others = [0, 1, 2, 3].filter((p) => !decl.includes(p));
  const t = decl.reduce((a, p) => a + tricks[p], 0);
  const team = (v) => { decl.forEach((p) => (pts[p] += v)); others.forEach((p) => (pts[p] -= v)); };
  const solo = (v) => { pts[decl[0]] += v; others.forEach((p) => (pts[p] -= v / 3)); };
  switch (ct.type) {
    case "vraag": team(vraagPts(t)); break;
    case "troel": team(troelPts(t)); break;
    case "troela": team(troelaPts(t)); break;
    case "alleen": solo(alleenPts(t)); break;
    case "abondance": solo(t >= ct.level ? ABON_PTS[ct.level] : -ABON_PTS[ct.level]); break;
    case "solo": solo(t === 13 ? 75 : -75); break;
    case "soloslim": solo(t === 13 ? 90 : -90); break;
    case "miserie": case "openmiserie": {
      const k = ct.type === "openmiserie" ? 2 : 1;
      const succ = decl.filter((p) => tricks[p] === 0).length;
      const [win, lose, oth] = miseryPts(decl.length, succ);
      decl.forEach((p) => (pts[p] += k * (tricks[p] === 0 ? win : lose)));
      others.forEach((p) => (pts[p] += k * oth));
      break;
    }
  }
  return pts;
}

// ------------------------------------------------------------------ bids
// Ranking low -> high (Reglement §2, troel "gaat boven alles behalve miserie op tafel, solo, solo-slim").
// Abondance of equal level: the one in the turned trump suit has priority.
function bidRank(b, turnedSuit) {
  switch (b.type) {
    case "vraag": return 10;
    case "abondance": return { 9: 20, 10: 30, 11: 50, 12: 60 }[b.level] + (b.trump === turnedSuit ? 1 : 0);
    case "miserie": return 40;
    case "troel": return 70;
    case "openmiserie": return 80;
    case "solo": return 90;
    case "soloslim": return 100;
  }
  return 0;
}
function bidLabel(b) {
  switch (b.type) {
    case "pas": return "Pas";
    case "vraag": return "Vraag";
    case "mee": return "Mee";
    case "alleen": return "Alleen";
    case "abondance": return `Abondance ${b.level} ${SUIT_SYM[b.trump]}`;
    case "miserie": return "Miserie";
    case "openmiserie": return "Miserie op tafel";
    case "troel": return "Troel";
    case "solo": return `Solo ${b.trump === NT ? "zonder troef" : SUIT_SYM[b.trump]}`;
    case "soloslim": return "Solo slim";
  }
  return b.type;
}

// ------------------------------------------------------------------ deck, dealing, collecting
function rngFrom(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
function shuffleInPlace(a, rand) {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
// deal positions -> seat. pattern "445" or "454"; dealing starts left of the dealer, the
// dealer receives the last card (the turned trump card, position 51).
function dealMap(dealer, pattern) {
  const sizes = pattern === "454" ? [4, 5, 4] : [4, 4, 5];
  const map = [];
  for (const sz of sizes) for (let k = 1; k <= 4; k++) for (let i = 0; i < sz; i++) map.push((dealer + k) % 4);
  return map;
}

// ------------------------------------------------------------------ game state
class Game {
  constructor(opts = {}) {
    this.rand = opts.rand || Math.random;
    this.scores = [0, 0, 0, 0];
    this.dealer = opts.dealer ?? 0;
    this.alwaysShuffle = !!opts.alwaysShuffle;
    this.deck = null;      // physical order (top first) for the next deal
    this.track = { kind: "shuffled" }; // public description of how `deck` was formed
    this.handNo = 0;
    this.history = [];     // finished hands
  }

  // --- start a new hand. cut in [4,48] (cards lifted off by the previous dealer), pattern "445"/"454"
  deal({ cut, pattern } = {}) {
    if (!this.deck || this.alwaysShuffle) {
      this.deck = shuffleInPlace([...Array(52).keys()], this.rand);
      this.track = { kind: "shuffled" };
    }
    if (cut == null) cut = 4 + Math.floor(this.rand() * 45);
    if (!pattern) pattern = this.rand() < 0.5 ? "445" : "454";
    const d = this.deck.slice(cut).concat(this.deck.slice(0, cut));
    const map = dealMap(this.dealer, pattern);
    this.hands = [[], [], [], []];
    d.forEach((c, i) => this.hands[map[i]].push(c));
    this.dealt = this.hands.map((h) => h.slice());
    this.turned = d[51];
    this.turnedSuit = suitOf(this.turned);
    this.pattern = pattern;
    this.cut = cut;           // secret: not part of the public information
    this.handTrack = this.track;
    this.phase = "bid";
    this.tricks = [];
    this.trick = [];
    this.won = [0, 0, 0, 0];
    this.contract = null;
    this.bids = [];
    this.openCards = null;
    // auction
    this.status = [0, 1, 2, 3].map(() => "in");
    this.cur = null;
    this.holders = [];
    this.askAlone = null;
    this.troelSeat = null;
    const aces = this.hands.map((h) => h.filter((c) => rankOf(c) === 12).length);
    const t = aces.findIndex((n) => n >= 3);
    if (t >= 0) {  // Troel MUST be announced; others may still go higher (miserie op tafel, solo, solo slim)
      this.troelSeat = t;
      this.cur = { type: "troel" };
      this.holders = [t];
      this.bids.push({ seat: t, bid: { type: "troel" } });
    }
    this.turn = this.nextBidder((this.dealer + 1) % 4, true);
    if (this.turn < 0) this.finishAuction();
  }

  curRank() { return this.cur ? bidRank(this.cur, this.turnedSuit) : 0; }

  nextBidder(from, inclusive) {
    for (let k = inclusive ? 0 : 1; k < 5; k++) {
      const p = (from + k) % 4;
      if (this.status[p] === "in" && !this.holders.includes(p)) return p;
    }
    return -1;
  }

  legalBids(seat) {
    if (this.phase !== "bid" || seat !== this.turn) return [];
    const out = [{ type: "pas" }];
    if (this.askAlone === seat) return out.concat([{ type: "alleen" }]);
    const r = this.curRank(), cur = this.cur;
    const ts = this.turnedSuit;
    const troel = cur && cur.type === "troel";
    if (!troel) {
      if (!cur) out.push({ type: "vraag" });
      if (cur && cur.type === "vraag" && this.holders.length === 1) out.push({ type: "mee" });
      for (const level of [9, 10, 11, 12])
        for (let t = 0; t < 4; t++) {
          const b = { type: "abondance", level, trump: t };
          if (bidRank(b, ts) > r) out.push(b);
        }
      if (r < 40 || (cur.type === "miserie")) out.push({ type: "miserie" });
    }
    if (r < 80 || (cur.type === "openmiserie")) out.push({ type: "openmiserie" });
    if (r < 90) for (let t = 0; t <= NT; t++) out.push({ type: "solo", trump: t });
    if (r < 100) out.push({ type: "soloslim" });
    return out;
  }

  bid(seat, b) {
    const legal = this.legalBids(seat);
    if (!legal.some((x) => JSON.stringify(x) === JSON.stringify(b))) throw new Error("illegal bid " + JSON.stringify(b));
    this.bids.push({ seat, bid: b });
    if (b.type === "pas") {
      if (this.askAlone === seat) { this.cur = null; this.holders = []; this.askAlone = null; this.status[seat] = "out"; return this.finishAuction(); }
      this.status[seat] = "out";
    } else if (b.type === "alleen") {
      this.cur = { type: "alleen" }; this.askAlone = null;
      return this.finishAuction();
    } else if (b.type === "mee" || (this.cur && b.type === this.cur.type && (b.type === "miserie" || b.type === "openmiserie"))) {
      this.holders.push(seat);
    } else {
      this.cur = b;
      this.holders = [seat];
    }
    this.turn = this.nextBidder(seat, false);
    if (this.turn < 0) {
      if (this.cur && this.cur.type === "vraag" && this.holders.length === 1) {
        // nobody joined: the asker may pass or play alone (5 tricks)
        this.askAlone = this.holders[0];
        this.turn = this.askAlone;
        this.holders = [];
        return;
      }
      this.finishAuction();
    }
  }

  finishAuction() {
    const ts = this.turnedSuit, voor = (this.dealer + 1) % 4;
    const cur = this.cur;
    if (!cur) { this.contract = null; this.phase = "allpass"; return this.endHand(); }
    let ct;
    const h = this.holders;
    switch (cur.type) {
      case "vraag": ct = { type: "vraag", players: h.slice(), trump: ts, leader: voor, target: 8 }; break;
      case "alleen": ct = { type: "alleen", players: [this.bids[this.bids.length - 1].seat], trump: ts, leader: voor, target: 5 }; break;
      case "abondance": ct = { type: "abondance", players: [h[0]], trump: cur.trump, leader: h[0], target: cur.level, level: cur.level }; break;
      case "miserie": case "openmiserie":
        ct = { type: cur.type, players: h.slice(), trump: NT, leader: voor, target: 0 }; break; // ASSUMPTION: no trump, voorhand leads
      case "solo": ct = { type: "solo", players: [h[0]], trump: cur.trump, leader: h[0], target: 13 }; break;
      case "soloslim": ct = { type: "soloslim", players: [h[0]], trump: ts, leader: h[0], target: 13 }; break;
      case "troel": {
        const T = this.troelSeat, mine = this.hands[T];
        const aces = mine.filter((c) => rankOf(c) === 12);
        let card;
        if (aces.length === 3) card = [0, 1, 2, 3].map((s) => s * 13 + 12).find((c) => !mine.includes(c));
        else { card = 2 * 13 + 11; while (mine.includes(card)) card--; } // troela: ♥H, else ♥V, ...
        const partner = this.hands.findIndex((hh) => hh.includes(card));
        ct = aces.length === 3
          ? { type: "troel", players: [T, partner], trump: suitOf(card), leader: partner, target: 8, forcedLead: card }
          : { type: "troela", players: [T, partner], trump: 2, leader: partner, target: 9, forcedLead: card };
        break;
      }
    }
    this.contract = ct;
    this.phase = "play";
    this.leader = ct.leader;
    this.turn = ct.leader;
    if (ct.type === "openmiserie") this.openCards = ct.players.slice();
  }

  legalCards(seat) {
    if (this.phase !== "play" || seat !== this.turn) return [];
    const hand = this.hands[seat];
    if (this.tricks.length === 0 && this.trick.length === 0 && this.contract.forcedLead != null) return [this.contract.forcedLead];
    if (this.trick.length) {
      const led = suitOf(this.trick[0].card);
      const f = hand.filter((c) => suitOf(c) === led);
      if (f.length) return f;
    }
    return hand.slice();
  }

  play(seat, card) {
    if (!this.legalCards(seat).includes(card)) throw new Error("illegal card " + cardStr(card));
    this.hands[seat] = this.hands[seat].filter((c) => c !== card);
    this.trick.push({ seat, card });
    if (this.trick.length < 4) { this.turn = (seat + 1) % 4; return null; }
    const w = trickWinner(this.trick, this.contract.trump);
    this.won[w]++;
    this.tricks.push({ cards: this.trick.map((x) => x.card), seats: this.trick.map((x) => x.seat), winner: w });
    this.trick = [];
    this.leader = w; this.turn = w;
    if (this.tricks.length === 13) this.endHand();
    return w;
  }

  endHand() {
    let pts = [0, 0, 0, 0];
    if (this.contract) pts = scoreHand(this.contract, this.won);
    for (let p = 0; p < 4; p++) this.scores[p] += pts[p];
    this.lastPts = pts;
    this.history.push({ dealer: this.dealer, contract: this.contract, won: this.won.slice(), pts, bids: this.bids.slice(), dealt: this.dealt });
    // --- collect the cards for the next deal (no shuffling after the first game, §1)
    const order = shuffleInPlace([0, 1, 2, 3], this.rand); // the new dealer gathers the piles in some order
    const deck = [];
    if (this.contract) {
      const piles = [[], [], [], []];
      for (const t of this.tricks) piles[t.winner].push(shuffleInPlace(t.cards.slice(), this.rand)); // order inside a trick is sloppy
      for (const p of order) for (const t of piles[p]) deck.push(...t);
      this.track = { kind: "tricks", tricks: this.tricks.map((t) => ({ cards: t.cards.slice(), winner: t.winner })) };
    } else {
      for (const p of order) deck.push(...shuffleInPlace(this.hands[p].slice(), this.rand));
      this.track = { kind: "hands" }; // each player only knows his own previous hand
      this.track.prevHands = this.hands.map((h) => h.slice()); // private to each seat; Belief uses only its own
    }
    this.deck = deck;
    this.phase = "done";
    this.handNo++;
    this.dealer = (this.dealer + 1) % 4; // ASSUMPTION: the deal passes on, also after "alle pas"
  }
}

function trickWinner(trick, trump) {
  let best = trick[0];
  for (let i = 1; i < trick.length; i++) {
    const c = trick[i].card, b = best.card;
    if ((suitOf(c) === suitOf(b) && c > b) || (suitOf(c) === trump && suitOf(b) !== trump)) best = trick[i];
  }
  return best.seat;
}

if (typeof module !== "undefined")
  module.exports = {
    SUIT_SYM, SUIT_NAME, RANK_STR, suitOf, rankOf, cardStr, NT, ABON_PTS, vraagPts, alleenPts, troelPts, troelaPts,
    miseryPts, scoreHand, bidRank, bidLabel, rngFrom, shuffleInPlace, dealMap, Game, trickWinner,
  };
