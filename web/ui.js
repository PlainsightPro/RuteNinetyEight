// ui.js — Wiezen table: you (south) against three bots. Requires the engine bundle (require()).
"use strict";
(function () {
  const R = require("rules");
  const { Bot } = require("ai");
  const { WiezenLM } = require("wiezenlm");
  const { Habits } = require("belief");
  const $ = (id) => document.getElementById(id);
  const NAMES = ["Jij", "Staf", "Mia", "Rik"];
  const POS = ["s", "w", "n", "e"];
  const SUIT_ORDER = [3, 2, 0, 1]; // ♠ ♥ ♣ ♦ (alternating colours)
  const weightsText = $("wiezenlm-weights").textContent;
  const opts = { tracking: true, shuffle: false, showAll: false };
  let game = null, worker = null, msgId = 0, busy = false, shownTrick = null, lastBotValues = null;
  let habits = new Habits(), habitsSeen = 0; // the same learning the bots do, shown in the "Gewoontes" panel
  const cbs = {};
  const fmt = (x) => (Math.round(x * 10) / 10).toString().replace(/\.0$/, "");
  const sgn = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x));

  // ------------------------------------------------------------ bots (Web Worker, fallback inline)
  const WORKER_TAIL = `
;(function(){
  const R = require("rules"), { Bot } = require("ai"), { WiezenLM } = require("wiezenlm");
  let bots = {};
  onmessage = (e) => {
    const d = e.data;
    if (d.init) {
      const lm = new WiezenLM(JSON.parse(d.weights));
      const prior = (h, t, s, loc) => lm.predict(h, t, s, loc); prior.usesLoc = lm.usesLoc;
      bots.on = new Bot({ prior, worlds: 32, bidWorlds: 14, timeMs: 1100 });
      bots.off = new Bot({ prior, worlds: 32, bidWorlds: 14, timeMs: 1100, useTracking: false });
      return;
    }
    const g = Object.assign(Object.create(R.Game.prototype), d.game);
    g.rand = Math.random;
    const b = d.tracking ? bots.on : bots.off;
    let out;
    try {
      if (d.kind === "bid") { const r = b.chooseBid(g, d.seat); out = { bid: r.bid, values: r.values || null }; }
      else { const r = b.choosePlay(g, d.seat); out = { card: r.card, values: r.values || null }; }
    } catch (err) { out = { error: String(err && err.stack || err) }; }
    postMessage({ id: d.id, res: out });
  };
})();`;
  let inline = null;
  function initBots() {
    try {
      const src = $("engine-src").textContent + WORKER_TAIL;
      worker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
      worker.onmessage = (e) => { const cb = cbs[e.data.id]; delete cbs[e.data.id]; if (cb) cb(e.data.res); };
      worker.onerror = () => { worker = null; };
      worker.postMessage({ init: true, weights: weightsText });
    } catch (e) { worker = null; }
  }
  function inlineBots() {
    if (!inline) {
      const lm = new WiezenLM(JSON.parse(weightsText));
      const prior = (h, t, s, loc) => lm.predict(h, t, s, loc); prior.usesLoc = lm.usesLoc;
      inline = { on: new Bot({ prior, worlds: 20, bidWorlds: 10, timeMs: 700 }), off: new Bot({ prior, worlds: 20, bidWorlds: 10, timeMs: 700, useTracking: false }) };
    }
    return opts.tracking ? inline.on : inline.off;
  }
  function snapshot() { return JSON.parse(JSON.stringify(game)); }
  function ask(kind, seat) {
    return new Promise((resolve) => {
      if (worker) {
        const id = ++msgId;
        cbs[id] = resolve;
        worker.postMessage({ id, kind, seat, game: snapshot(), tracking: opts.tracking });
        setTimeout(() => { if (cbs[id]) { delete cbs[id]; worker = null; resolve(localDecide(kind, seat)); } }, 20000);
      } else setTimeout(() => resolve(localDecide(kind, seat)), 30);
    });
  }
  function localDecide(kind, seat) {
    const b = inlineBots();
    return kind === "bid" ? b.chooseBid(game, seat) : b.choosePlay(game, seat);
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ------------------------------------------------------------ game flow
  function newGame() {
    game = new R.Game({ dealer: 1, alwaysShuffle: opts.shuffle });
    habits = new Habits(); habitsSeen = 0;
    lastBotValues = null;
    prepareDeal();
  }
  function prepareDeal() {
    game.alwaysShuffle = opts.shuffle;
    const cutter = (game.dealer + 3) % 4;
    shownTrick = null;
    renderAll();
    if (cutter === 0 || game.dealer === 0) { showDealPanel(cutter === 0, game.dealer === 0); return; }
    doDeal({});
  }
  function doDeal({ cut, pattern }) {
    $("deal-panel").hidden = true;
    game.deal({ cut, pattern });
    lastBotValues = null;
    renderAll();
    step();
  }

  async function step() {
    if (busy) return;
    renderAll();
    if (game.phase === "bid" && game.turn !== 0) {
      busy = true;
      $("thinking").hidden = false;
      const t0 = Date.now();
      const r = await ask("bid", game.turn);
      await sleep(Math.max(0, 550 - (Date.now() - t0)));
      $("thinking").hidden = true;
      busy = false;
      if (r.error) { console.error(r.error); return; }
      lastBotValues = { seat: game.turn, values: r.values };
      game.bid(game.turn, r.bid);
      return step();
    }
    if (game.phase === "play" && game.turn !== 0) {
      busy = true;
      $("thinking").hidden = false;
      const t0 = Date.now();
      const r = await ask("play", game.turn);
      await sleep(Math.max(0, 450 - (Date.now() - t0)));
      $("thinking").hidden = true;
      busy = false;
      if (r.error) { console.error(r.error); return; }
      await playCard(game.turn, r.card);
      return step();
    }
  }

  async function playCard(seat, card) {
    const n = game.tricks.length;
    game.play(seat, card);
    if (game.tricks.length > n) {  // trick complete: leave it on the table for a moment
      shownTrick = game.tricks[game.tricks.length - 1];
      renderAll();
      busy = true;
      await sleep(1100);
      busy = false;
      shownTrick = null;
    }
    renderAll();
  }

  function humanBid(b) { if (busy || game.phase !== "bid" || game.turn !== 0) return; game.bid(0, b); step(); }
  async function humanPlay(card) {
    if (busy || game.phase !== "play" || game.turn !== 0 || !game.legalCards(0).includes(card)) return;
    await playCard(0, card);
    step();
  }

  // ------------------------------------------------------------ rendering
  function cardEl(c, cls = "") {
    const el = document.createElement("div");
    if (c == null) { el.className = "card back " + cls; return el; }
    const s = R.suitOf(c);
    el.className = `card ${s === 1 || s === 2 ? "red" : ""} ${cls}`;
    el.innerHTML = `<span class="ix"><span class="rk">${R.RANK_STR[R.rankOf(c)]}</span><span class="st">${R.SUIT_SYM[s]}</span></span><span class="pip">${R.SUIT_SYM[s]}</span>`;
    el.setAttribute("aria-label", R.cardStr(c));
    return el;
  }
  const sortHand = (h) => h.slice().sort((a, b) => SUIT_ORDER.indexOf(R.suitOf(a)) - SUIT_ORDER.indexOf(R.suitOf(b)) || R.rankOf(b) - R.rankOf(a));

  function roleOf(p) {
    const ct = game.contract;
    if (!ct) return "";
    if (!ct.players.includes(p)) return "";
    if (ct.type === "miserie" || ct.type === "openmiserie") return "miserie";
    return ct.players[0] === p ? "speler" : "partner";
  }
  function contractText(ct) {
    if (!ct) return game.phase === "bid" ? "Bieden…" : "Alle pas";
    const who = ct.players.map((p) => NAMES[p]).join(" + ");
    const tr = ct.trump === R.NT ? "zonder troef" : `troef ${R.SUIT_SYM[ct.trump]}`;
    const name = { vraag: "Vragen en meegaan", alleen: "Alleen", abondance: `Abondance ${ct.level}`, miserie: "Miserie",
      openmiserie: "Miserie op tafel", troel: "Troel", troela: "Troela", solo: "Solo", soloslim: "Solo slim" }[ct.type];
    const goal = ct.type.includes("miserie") ? "geen slag halen" : `${ct.target} slagen`;
    return `${name} · ${who} · ${tr} · ${goal}`;
  }

  function renderAll() {
    if (!game) return;
    const over = game.phase === "done" || game.phase === "allpass";
    // scores
    $("scores").replaceChildren(...[0, 1, 2, 3].map((p) => {
      const d = document.createElement("div");
      d.className = "sc";
      const last = game.lastPts && over ? `<span class="delta ${game.lastPts[p] >= 0 ? "pos" : "neg"}">${sgn(game.lastPts[p])}</span>` : "";
      d.innerHTML = `<span class="k">${NAMES[p]}</span><span class="v">${fmt(game.scores[p])}</span>${last}`;
      return d;
    }));
    $("handno").textContent = `Spel ${game.handNo + (over ? 0 : 1)}`;
    if (!game.hands) { $("contract").textContent = ""; return; }
    // seats
    for (let p = 0; p < 4; p++) {
      const el = $("seat-" + POS[p]);
      el.querySelector(".nm").textContent = NAMES[p];
      el.querySelector(".dl").hidden = handDealer() !== p;
      const role = roleOf(p);
      const rb = el.querySelector(".role");
      rb.textContent = role; rb.hidden = !role; rb.className = "role " + role;
      el.querySelector(".won").textContent = game.phase === "play" || over ? `${game.won[p]} slag${game.won[p] === 1 ? "" : "en"}` : "";
      const lastBid = [...game.bids].reverse().find((b) => b.seat === p);
      el.querySelector(".bub").textContent = game.phase === "bid" || (!game.contract && over) ? (lastBid ? R.bidLabel(lastBid.bid) : "") : "";
      el.classList.toggle("turn", !over && game.turn === p && !shownTrick);
      if (p !== 0) {
        const cards = el.querySelector(".cards");
        const open = opts.showAll || over || (game.openCards && game.openCards.includes(p) && game.phase === "play");
        const hand = over ? game.dealt[p] : game.hands[p];
        cards.replaceChildren(...(open ? sortHand(hand).map((c) => cardEl(c, "sm")) : hand.map(() => cardEl(null, "sm"))));
        cards.classList.toggle("open", !!open);
      }
    }
    // your hand
    const legal = game.phase === "play" && game.turn === 0 && !busy ? new Set(game.legalCards(0)) : new Set();
    const myHand = over ? game.dealt[0] : game.hands[0];
    $("hand").replaceChildren(...sortHand(myHand).map((c) => {
      const el = cardEl(c, legal.size ? (legal.has(c) ? "legal" : "dim") : "");
      if (legal.has(c)) { el.tabIndex = 0; el.onclick = () => humanPlay(c); el.onkeydown = (e) => { if (e.key === "Enter") humanPlay(c); }; }
      return el;
    }));
    // trick
    const tr = shownTrick ? shownTrick.cards.map((c, i) => ({ seat: shownTrick.seats[i], card: c })) : game.trick;
    for (let p = 0; p < 4; p++) {
      const slot = $("t-" + POS[p]);
      const x = tr.find((y) => y.seat === p);
      slot.replaceChildren(...(x ? [cardEl(x.card, shownTrick && shownTrick.winner === p ? "win" : "")] : []));
    }
    // turned card + contract
    const turnedPlayed = game.tricks.some((t) => t.cards.includes(game.turned)) || game.trick.some((x) => x.card === game.turned);
    $("turned").replaceChildren(...(turnedPlayed && !over ? [] : [cardEl(game.turned, "sm")]));
    $("turned-who").textContent = turnedPlayed && !over
      ? `Troef ${game.contract && game.contract.trump < 4 ? R.SUIT_SYM[game.contract.trump] : "–"} · gever ${NAMES[handDealer()]}`
      : `Keerkaart · gever ${NAMES[handDealer()]}`;
    $("contract").textContent = contractText(game.contract);
    $("deck-info").textContent = trackText();
    renderBidPanel();
    renderLog();
    renderPrev();
    renderWhy();
    renderHabits();
    // end of hand
    const sum = $("summary");
    if (over) {
      sum.hidden = false;
      const ct = game.contract;
      if (!ct) $("sum-text").textContent = "Iedereen past. Het volgende spel wordt gegeven zonder te schudden.";
      else {
        const t = ct.players.reduce((a, p) => a + game.won[p], 0);
        const ok = ct.type.includes("miserie") ? ct.players.filter((p) => game.won[p] === 0).length + " van " + ct.players.length + " miserie gehaald"
          : t >= ct.target ? `gehaald met ${t} slagen` : `niet gehaald: ${t} slagen`;
        $("sum-text").textContent = `${contractText(ct)}: ${ok}.`;
      }
    } else sum.hidden = true;
  }

  // after a hand, game.dealer already points at the next dealer
  function handDealer() {
    const over = game.phase === "done" || game.phase === "allpass";
    return over && game.history.length ? game.history[game.history.length - 1].dealer : game.dealer;
  }

  function trackText() {
    const k = game.handTrack ? game.handTrack.kind : "shuffled";
    const pat = game.pattern ? game.pattern.split("").join("-") : "";
    if (k === "shuffled") return `Geschud deck · gegeven ${pat}`;
    if (k === "tricks") return `Niet geschud: slagen van vorig spel opgeraapt, gecoupeerd, gegeven ${pat}`;
    return `Niet geschud: handen van het passpel opgeraapt, gegeven ${pat}`;
  }

  function renderBidPanel() {
    const panel = $("bid-panel");
    const mine = game.phase === "bid" && game.turn === 0 && !busy;
    panel.hidden = !mine;
    if (!mine) return;
    const legal = game.legalBids(0);
    const has = (f) => legal.filter(f);
    const btn = (b, label, cls = "") => {
      const x = document.createElement("button");
      x.textContent = label || R.bidLabel(b);
      x.className = cls;
      x.onclick = () => humanBid(b);
      return x;
    };
    const main = $("bid-main");
    main.replaceChildren(...legal.filter((b) => ["pas", "vraag", "mee", "alleen", "miserie", "openmiserie", "soloslim"].includes(b.type))
      .map((b) => btn(b, null, b.type === "pas" ? "ghost" : "")));
    const ab = $("bid-abon");
    ab.replaceChildren();
    for (const lv of [9, 10, 11, 12]) {
      const opts2 = has((b) => b.type === "abondance" && b.level === lv);
      if (!opts2.length) continue;
      const row = document.createElement("div");
      row.className = "brow";
      row.innerHTML = `<span class="bl">Abondance ${lv}</span>`;
      opts2.forEach((b) => row.appendChild(btn(b, R.SUIT_SYM[b.trump], "suit " + (b.trump === 1 || b.trump === 2 ? "red" : ""))));
      ab.appendChild(row);
    }
    const solos = has((b) => b.type === "solo");
    if (solos.length) {
      const row = document.createElement("div");
      row.className = "brow";
      row.innerHTML = `<span class="bl">Solo</span>`;
      solos.forEach((b) => row.appendChild(btn(b, b.trump === R.NT ? "SA" : R.SUIT_SYM[b.trump], "suit " + (b.trump === 1 || b.trump === 2 ? "red" : ""))));
      ab.appendChild(row);
    }
    $("bid-hint").textContent = game.askAlone === 0 ? "Niemand ging mee. Speel je alleen (5 slagen) of pas je?" : "Jouw bod";
  }

  function renderLog() {
    const items = [];
    for (const b of game.bids) items.push({ cls: b.seat === 0 ? "me" : "", t: `${NAMES[b.seat]}: ${R.bidLabel(b.bid)}` });
    game.tricks.forEach((t, i) => items.push({ cls: "tr", t: `Slag ${i + 1}: ${t.cards.map((c, j) => `${NAMES[t.seats[j]][0]} ${R.cardStr(c)}`).join("  ")} → ${NAMES[t.winner]}` }));
    $("log").replaceChildren(...items.map((x) => { const li = document.createElement("li"); li.className = x.cls; li.textContent = x.t; return li; }));
    $("log").scrollTop = $("log").scrollHeight;
  }

  function renderPrev() {
    const h = game.handTrack;
    const box = $("prev");
    if (!h || h.kind !== "tricks") {
      box.innerHTML = `<p class="note">${!h || h.kind === "shuffled" ? "Eerste spel: het deck is geschud, er valt niets te volgen." : "Vorig spel was alle pas: alleen je eigen vorige hand is bekend."}</p>`;
      return;
    }
    const piles = [[], [], [], []];
    h.tricks.forEach((t, i) => piles[t.winner].push({ i, cards: t.cards }));
    box.replaceChildren(...[0, 1, 2, 3].filter((p) => piles[p].length).map((p) => {
      const d = document.createElement("div");
      d.className = "pile";
      d.innerHTML = `<span class="pn">${NAMES[p]}</span>` + piles[p].map((t) => `<span class="pt">${t.cards.map((c) => `<b class="${[1, 2].includes(R.suitOf(c)) ? "red" : ""}">${R.cardStr(c)}</b>`).join("")}</span>`).join("");
      return d;
    }));
  }

  function renderWhy() {
    const box = $("why");
    if (!lastBotValues || !lastBotValues.values) { box.textContent = "Na elk bod van een bot zie je hier hoe het de opties inschatte."; return; }
    const v = lastBotValues.values.filter((x) => x.bid.type !== "pas").sort((a, b) => b.ev - a.ev).slice(0, 5);
    box.replaceChildren(...[`${NAMES[lastBotValues.seat]} rekende (verwachte punten):`, ...v.map((x) => `${R.bidLabel(x.bid)}: ${sgn(x.ev)}`), "Pas: 0"]
      .map((t, i) => { const d = document.createElement("div"); d.textContent = t; if (i === 0) d.className = "wh"; return d; }));
  }

  function renderHabits() {
    for (; habitsSeen < game.history.length; habitsSeen++) habits.learnFrom(game.history[habitsSeen]);
    const box = $("habits");
    if (!habits.learned) { box.innerHTML = `<p class="note">Na elk niet-geschud spel reconstrueren de bots hoe er werd opgeraapt en gecoupeerd, en leren ze ieders gewoontes.</p>`; return; }
    const rows = [0, 1, 2, 3].map((p) => {
      const a = habits.cut[p]; let t = 0, m = 0;
      for (let c = 4; c <= 48; c++) { t += a[c]; m += c * a[c]; }
      const neat = 1 - habits.pickMix(p)[26];
      const pick = p === 0 ? "" : ` · slagen ${neat > 0.6 ? "netjes" : neat > 0.4 ? "half netjes" : "slordig"} opgeraapt`;
      return `<div><b>${NAMES[p]}</b> coupeert rond ${Math.round(m / t)} kaarten${pick}</div>`;
    });
    box.innerHTML = rows.join("") + `<p class="note">Geleerd uit ${habits.learned} spel${habits.learned === 1 ? "" : "len"}. Jouw afpakken is de gewoonte die jij zelf kiest: wie altijd ongeveer even diep coupeert, maakt het de bots makkelijker.</p>`;
  }

  function showDealPanel(iCut, iDeal) {
    const p = $("deal-panel");
    p.hidden = false;
    $("cut-row").hidden = !iCut;
    $("pattern-row").hidden = !iDeal;
    $("deal-title").textContent = iCut && iDeal ? "Jij coupeert en geeft" : iCut ? `Jij coupeert voor ${NAMES[game.dealer]}` : "Jij geeft";
    $("cut-val").textContent = $("cut").value;
  }

  // ------------------------------------------------------------ wiring
  $("cut").oninput = () => ($("cut-val").textContent = $("cut").value);
  $("deal-go").onclick = () => {
    const iCut = !$("cut-row").hidden, iDeal = !$("pattern-row").hidden;
    doDeal({ cut: iCut ? +$("cut").value : undefined, pattern: iDeal ? document.querySelector("input[name=pattern]:checked").value : undefined });
  };
  $("next").onclick = () => prepareDeal();
  $("restart").onclick = () => { if (!busy) newGame(); };
  $("opt-track").onchange = (e) => (opts.tracking = e.target.checked);
  $("opt-shuffle").onchange = (e) => (opts.shuffle = e.target.checked);
  $("opt-open").onchange = (e) => { opts.showAll = e.target.checked; renderAll(); };

  initBots();
  newGame();
  window.__wiezen = { get game() { return game; }, humanBid, humanPlay, get busy() { return busy; }, doDeal, prepareDeal };
})();
