// Pure-JS reference forward pass for Q-Omni (CPU). Mirrors model/qpack_run.py (NumpyQOmni) exactly:
// same op order, same GQA head grouping, same RoPE, same per-head output gate, same 2-matrix SiLU MLP.
// This is the fallback path (and the correctness oracle for the WebGPU path in gpu.js).
'use strict';

function linear(x, W, outDim, inDim) { // y = x @ W^T ; W flat row-major [outDim, inDim]
  const y = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let s = 0; const base = o * inDim;
    for (let k = 0; k < inDim; k++) s += x[k] * W[base + k];
    y[o] = s;
  }
  return y;
}

function rmsnorm(x, w, eps) {
  let ss = 0; for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / x.length + eps);
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i] * inv * w[i];
  return y;
}

function silu(v) { return v / (1 + Math.exp(-v)); }
function sigmoid(v) { return 1 / (1 + Math.exp(-v)); }

class QOmniJS {
  constructor(loaded) {
    this.cfg = loaded.cfg; this.t = loaded.tensors; this.table = loaded.table;
    const c = this.cfg;
    this.nh = c.num_attention_heads; this.nkv = c.num_key_value_heads; this.hd = c.head_dim;
    this.d = c.hidden_size; this.ff = c.intermediate_size; this.L = c.num_hidden_layers;
    this.eps = c.rms_norm_eps; this.bits = c.code_bits;
    const invFreq = new Float32Array(this.hd / 2);
    for (let i = 0; i < invFreq.length; i++) invFreq[i] = 1 / Math.pow(c.rope_theta, (2 * i) / this.hd);
    this.invFreq = invFreq;
    this.reset();
  }

  reset() { // fresh KV cache
    this.kCache = []; this.vCache = [];
    for (let l = 0; l < this.L; l++) { this.kCache.push([]); this.vCache.push([]); } // each: array of Float32Array[nkv*hd] per position
    this.pos = 0;
  }

  ropeAt(pos) {
    const hd = this.hd, cos = new Float32Array(hd), sin = new Float32Array(hd);
    for (let i = 0; i < hd / 2; i++) { const a = pos * this.invFreq[i]; const cv = Math.cos(a), sv = Math.sin(a); cos[i] = cv; cos[i + hd / 2] = cv; sin[i] = sv; sin[i + hd / 2] = sv; }
    return { cos, sin };
  }

  applyRope(vec, cos, sin) { // vec length hd, rotate_half: [-x2, x1] where x1=first half, x2=second half
    const hd = vec.length, half = hd / 2, out = new Float32Array(hd);
    for (let i = 0; i < half; i++) {
      out[i] = vec[i] * cos[i] - vec[i + half] * sin[i];
      out[i + half] = vec[i + half] * cos[i + half] + vec[i] * sin[i + half];
    }
    return out;
  }

  embed(id) {
    const bits = this.bits, row = this.table.subarray(id * bits, id * bits + bits);
    return linear(row, this.t['model.w_in.weight'].data, this.d, bits);
  }

  logits(x) {
    const z = linear(x, this.t['w_out.weight'].data, this.bits, this.d);
    const vocab = this.cfg.vocab_size, out = new Float32Array(vocab), scale = 1 / Math.sqrt(this.bits);
    const bias = this.t['logit_bias'].data;
    for (let v = 0; v < vocab; v++) {
      let s = 0; const base = v * this.bits;
      for (let k = 0; k < this.bits; k++) s += z[k] * this.table[base + k];
      out[v] = s * scale + bias[v];
    }
    return out;
  }

  // one incremental decode step: token id -> logits over the vocab. Uses & grows the KV cache (causal by construction).
  step(id) {
    let x = this.embed(id);
    const { cos, sin } = this.ropeAt(this.pos);
    const rep = this.nh / this.nkv;
    for (let l = 0; l < this.L; l++) {
      const P = `model.layers.${l}.`;
      const h = rmsnorm(x, this.t[P + 'n1.weight'].data, this.eps);
      const qFlat = linear(h, this.t[P + 'attn.q_proj.weight'].data, this.nh * this.hd, this.d);
      const kFlat = linear(h, this.t[P + 'attn.k_proj.weight'].data, this.nkv * this.hd, this.d);
      const vFlat = linear(h, this.t[P + 'attn.v_proj.weight'].data, this.nkv * this.hd, this.d);
      const qn = this.t[P + 'attn.q_norm.weight'].data, kn = this.t[P + 'attn.k_norm.weight'].data;
      const qHeads = [], kHeads = [], vHeads = [];
      for (let hI = 0; hI < this.nh; hI++) qHeads.push(this.applyRope(rmsnorm(qFlat.subarray(hI * this.hd, (hI + 1) * this.hd), qn, this.eps), cos, sin));
      for (let hI = 0; hI < this.nkv; hI++) {
        kHeads.push(this.applyRope(rmsnorm(kFlat.subarray(hI * this.hd, (hI + 1) * this.hd), kn, this.eps), cos, sin));
        vHeads.push(vFlat.slice(hI * this.hd, (hI + 1) * this.hd));
      }
      for (let hI = 0; hI < this.nkv; hI++) { this.kCache[l][hI] = this.kCache[l][hI] || []; this.vCache[l][hI] = this.vCache[l][hI] || []; this.kCache[l][hI].push(kHeads[hI]); this.vCache[l][hI].push(vHeads[hI]); }
      const attnOut = new Float32Array(this.nh * this.hd);
      const gateLogit = linear(h, this.t[P + 'attn.out_gate.weight'].data, this.nh, this.d);
      const gateBias = this.t[P + 'attn.out_gate.bias'].data;
      for (let hI = 0; hI < this.nh; hI++) {
        const kv = (hI / rep) | 0; const kc = this.kCache[l][kv], vc = this.vCache[l][kv]; const T = kc.length;
        const scores = new Float32Array(T); const scale = 1 / Math.sqrt(this.hd);
        let mx = -Infinity;
        for (let ti = 0; ti < T; ti++) { let s = 0; for (let k = 0; k < this.hd; k++) s += qHeads[hI][k] * kc[ti][k]; s *= scale; scores[ti] = s; if (s > mx) mx = s; }
        let sum = 0; for (let ti = 0; ti < T; ti++) { scores[ti] = Math.exp(scores[ti] - mx); sum += scores[ti]; }
        const o = new Float32Array(this.hd);
        for (let ti = 0; ti < T; ti++) { const w = scores[ti] / sum; for (let k = 0; k < this.hd; k++) o[k] += w * vc[ti][k]; }
        const gate = sigmoid(gateLogit[hI] + gateBias[hI]);
        for (let k = 0; k < this.hd; k++) attnOut[hI * this.hd + k] = o[k] * gate;
      }
      const o = linear(attnOut, this.t[P + 'attn.o_proj.weight'].data, this.d, this.nh * this.hd);
      for (let k = 0; k < this.d; k++) x[k] += o[k];
      const h2 = rmsnorm(x, this.t[P + 'n2.weight'].data, this.eps);
      const u = linear(h2, this.t[P + 'mlp.w1.weight'].data, this.ff, this.d);
      for (let k = 0; k < this.ff; k++) u[k] = silu(u[k]);
      const mo = linear(u, this.t[P + 'mlp.w2.weight'].data, this.d, this.ff);
      for (let k = 0; k < this.d; k++) x[k] += mo[k];
    }
    x = rmsnorm(x, this.t['model.norm.weight'].data, this.eps);
    this.pos += 1;
    return this.logits(x);
  }
}



window.QOmniLib = window.QOmniLib || {};
window.QOmniLib.QOmniJS = QOmniJS;
