// wiezenlm.js — browser/Node inference for the WiezenLM hand evaluator (see train/train_wiezenlm.py).
"use strict";

function wlB64(b64) {
  if (typeof Buffer !== "undefined") return new Uint8Array(Buffer.from(b64, "base64"));
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function wlF16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

class WiezenLM {
  constructor(bundle) {
    this.cfg = bundle.config;
    this.W = {};
    for (const [k, v] of Object.entries(bundle.weights)) {
      const b = wlB64(v.b64), u = new Uint16Array(b.buffer, b.byteOffset, b.byteLength / 2);
      const a = new Float32Array(u.length);
      for (let i = 0; i < u.length; i++) a[i] = wlF16(u[i]);
      this.W[k] = a;
    }
    this.tests = bundle.tests;
    this.usesLoc = this.cfg.version === 2; // v2 also reads the card-tracking belief
  }
  static ln(x, T, d, w, b) {
    const y = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      let m = 0, v = 0;
      for (let i = 0; i < d; i++) m += x[t * d + i];
      m /= d;
      for (let i = 0; i < d; i++) { const z = x[t * d + i] - m; v += z * z; }
      const inv = 1 / Math.sqrt(v / d + 1e-5);
      for (let i = 0; i < d; i++) y[t * d + i] = (x[t * d + i] - m) * inv * w[i] + b[i];
    }
    return y;
  }
  static lin(x, T, din, dout, W, b) {
    const y = new Float32Array(T * dout);
    for (let t = 0; t < T; t++) {
      const xo = t * din, yo = t * dout;
      for (let j = 0; j < dout; j++) {
        let s0 = b[j], s1 = 0, s2 = 0, s3 = 0, wo = j * din, i = 0;
        for (; i + 3 < din; i += 4) {
          s0 += x[xo + i] * W[wo + i]; s1 += x[xo + i + 1] * W[wo + i + 1];
          s2 += x[xo + i + 2] * W[wo + i + 2]; s3 += x[xo + i + 3] * W[wo + i + 3];
        }
        for (; i < din; i++) s0 += x[xo + i] * W[wo + i];
        y[yo + j] = s0 + s1 + s2 + s3;
      }
    }
    return y;
  }
  // loc (v2 only): Float32Array(52*3), P(card held by the player at +1/+2/+3 seats)
  raw(cards, turned, seat, loc) {
    const { d, heads, layers } = this.cfg, W = this.W, dh = d / heads, v2 = this.usesLoc;
    const mine = new Set(cards);
    const others = v2 ? [...Array(52).keys()].filter((c) => !mine.has(c)) : [];
    const T = 16 + others.length;
    const x = new Float32Array(T * d);
    const kind = W["kind.weight"];
    const emb = (c, row) => { for (let i = 0; i < d; i++) x[row * d + i] = W["card.weight"][c * d + i] + W["suit.weight"][((c / 13) | 0) * d + i] + W["rank.weight"][(c % 13) * d + i]; };
    for (let i = 0; i < d; i++) x[i] = W["cls"][i] + kind[i];
    cards.forEach((c, j) => { emb(c, 1 + j); for (let i = 0; i < d; i++) x[(1 + j) * d + i] += kind[d + i]; });
    emb(turned, 14); for (let i = 0; i < d; i++) x[14 * d + i] += kind[2 * d + i];
    for (let i = 0; i < d; i++) x[15 * d + i] = W["seat.weight"][seat * d + i] + kind[3 * d + i];
    others.forEach((c, j) => {
      const row = 16 + j;
      emb(c, row);
      const l0 = loc ? loc[c * 3] : 0, l1 = loc ? loc[c * 3 + 1] : 0, l2 = loc ? loc[c * 3 + 2] : 0;
      for (let i = 0; i < d; i++)
        x[row * d + i] += kind[4 * d + i] + W["loc.bias"][i] + W["loc.weight"][i * 3] * l0 + W["loc.weight"][i * 3 + 1] * l1 + W["loc.weight"][i * 3 + 2] * l2;
    });
    let h = x;
    for (let l = 0; l < layers; l++) {
      const P = `blocks.${l}.`;
      const a = WiezenLM.ln(h, T, d, W[P + "ln1.weight"], W[P + "ln1.bias"]);
      const qkv = WiezenLM.lin(a, T, d, 3 * d, W[P + "qkv.weight"], W[P + "qkv.bias"]);
      const y = new Float32Array(T * d), sc = new Float32Array(T), scale = 1 / Math.sqrt(dh);
      const d3 = 3 * d;
      for (let hd = 0; hd < heads; hd++) {
        const ho = hd * dh;
        for (let t = 0; t < T; t++) {
          const qo = t * d3 + ho;
          let mx = -Infinity;
          for (let s = 0; s < T; s++) {
            const ko = s * d3 + d + ho;
            let dot = 0;
            for (let i = 0; i < dh; i++) dot += qkv[qo + i] * qkv[ko + i];
            dot *= scale; sc[s] = dot; if (dot > mx) mx = dot;
          }
          let z = 0;
          for (let s = 0; s < T; s++) { const e = Math.exp(sc[s] - mx); sc[s] = e; z += e; }
          const yo = t * d + ho, iz = 1 / z;
          for (let s = 0; s < T; s++) {
            const p = sc[s] * iz, vo = s * d3 + 2 * d + ho;
            for (let i = 0; i < dh; i++) y[yo + i] += p * qkv[vo + i];
          }
        }
      }
      const pr = WiezenLM.lin(y, T, d, d, W[P + "proj.weight"], W[P + "proj.bias"]);
      for (let i = 0; i < h.length; i++) h[i] += pr[i];
      const a2 = WiezenLM.ln(h, T, d, W[P + "ln2.weight"], W[P + "ln2.bias"]);
      const m = WiezenLM.lin(a2, T, d, 4 * d, W[P + "fc1.weight"], W[P + "fc1.bias"]);
      for (let i = 0; i < m.length; i++) { const v = m[i]; m[i] = 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))); }
      const o = WiezenLM.lin(m, T, 4 * d, d, W[P + "fc2.weight"], W[P + "fc2.bias"]);
      for (let i = 0; i < h.length; i++) h[i] += o[i];
    }
    const cls = WiezenLM.ln(h.subarray(0, d), 1, d, W["ln.weight"], W["ln.bias"]);
    return WiezenLM.lin(cls, 1, d, 127, W["head.weight"], W["head.bias"]);
  }
  // distributions in the format Bot.estimate() expects
  predict(cards, turned, seat, loc) {
    const o = this.raw(cards, turned, seat, loc);
    const dist = (k) => {
      let mx = -Infinity; for (let i = 0; i < 14; i++) mx = Math.max(mx, o[k * 14 + i]);
      const e = []; let z = 0; for (let i = 0; i < 14; i++) { e.push(Math.exp(o[k * 14 + i] - mx)); z += e[i]; }
      return e.map((v) => v / z);
    };
    return {
      alone: [dist(0), dist(1), dist(2), dist(3)], nt: dist(4), alleen: dist(5),
      pair: [dist(6), dist(7), dist(8)], mis: 1 / (1 + Math.exp(-o[126])),
    };
  }
  selfTest() {
    let worst = 0;
    for (const t of this.tests) {
      const o = this.raw(t.cards, t.turned, t.seat, t.loc ? Float32Array.from(t.loc) : null);
      for (let i = 0; i < 127; i++) worst = Math.max(worst, Math.abs(o[i] - t.out[i]));
    }
    return worst;
  }
}

if (typeof module !== "undefined") module.exports = { WiezenLM };
