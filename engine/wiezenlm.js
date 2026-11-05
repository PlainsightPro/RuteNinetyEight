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
    for (let t = 0; t < T; t++)
      for (let j = 0; j < dout; j++) {
        let s = b[j];
        for (let i = 0; i < din; i++) s += x[t * din + i] * W[j * din + i];
        y[t * dout + j] = s;
      }
    return y;
  }
  raw(cards, turned, seat) {
    const { d, heads, layers } = this.cfg, W = this.W, T = 16, dh = d / heads;
    const x = new Float32Array(T * d);
    const kind = W["kind.weight"];
    const emb = (c, row) => { for (let i = 0; i < d; i++) x[row * d + i] = W["card.weight"][c * d + i] + W["suit.weight"][((c / 13) | 0) * d + i] + W["rank.weight"][(c % 13) * d + i]; };
    for (let i = 0; i < d; i++) x[i] = W["cls"][i] + kind[i];
    cards.forEach((c, j) => { emb(c, 1 + j); for (let i = 0; i < d; i++) x[(1 + j) * d + i] += kind[d + i]; });
    emb(turned, 14); for (let i = 0; i < d; i++) x[14 * d + i] += kind[2 * d + i];
    for (let i = 0; i < d; i++) x[15 * d + i] = W["seat.weight"][seat * d + i] + kind[3 * d + i];
    let h = x;
    for (let l = 0; l < layers; l++) {
      const P = `blocks.${l}.`;
      const a = WiezenLM.ln(h, T, d, W[P + "ln1.weight"], W[P + "ln1.bias"]);
      const qkv = WiezenLM.lin(a, T, d, 3 * d, W[P + "qkv.weight"], W[P + "qkv.bias"]);
      const y = new Float32Array(T * d), sc = new Float32Array(T), scale = 1 / Math.sqrt(dh);
      for (let hd = 0; hd < heads; hd++)
        for (let t = 0; t < T; t++) {
          let mx = -Infinity;
          for (let s = 0; s < T; s++) {
            let dot = 0;
            for (let i = 0; i < dh; i++) dot += qkv[t * 3 * d + hd * dh + i] * qkv[s * 3 * d + d + hd * dh + i];
            sc[s] = dot * scale; if (sc[s] > mx) mx = sc[s];
          }
          let z = 0;
          for (let s = 0; s < T; s++) { sc[s] = Math.exp(sc[s] - mx); z += sc[s]; }
          for (let s = 0; s < T; s++) { const p = sc[s] / z; for (let i = 0; i < dh; i++) y[t * d + hd * dh + i] += p * qkv[s * 3 * d + 2 * d + hd * dh + i]; }
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
  predict(cards, turned, seat) {
    const o = this.raw(cards, turned, seat);
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
      const o = this.raw(t.cards, t.turned, t.seat);
      for (let i = 0; i < 127; i++) worst = Math.max(worst, Math.abs(o[i] - t.out[i]));
    }
    return worst;
  }
}

if (typeof module !== "undefined") module.exports = { WiezenLM };
