// N-Q (qwen4 family) forward pass in plain JavaScript, one token at a time. Mirrors nq/model/modeling_nq4.py
// (NQ4ForCausalLM, dense non-ternary checkpoints such as N-Q-26M-Chat) op for op:
//   embed -> 4 residual streams -> 8 x [ (layer 2: + n-gram PLE) -> Gated Residual -> Gated DeltaNet | gated attention
//   -> Gated Residual -> SwiGLU ] -> Gated Residual collapse -> lm_head.
// Big matrices can live in a WASM SIMD kernel (nq_kernel.wasm, attachKernel); weights may be f32/f16 (QPACK1) or int8 'q8' (QPACK2).
// Decode-time state per layer: Gated DeltaNet = short-conv window + recurrent matrix S per value head; attention =
// K/V cache; PLE = last two token ids + dilated conv window. Checked against PyTorch logits in node (test_parity.mjs).
'use strict';

(function () {
  const Lib = (typeof window !== 'undefined' ? (window.QOmniLib = window.QOmniLib || {}) : (globalThis.QOmniLib = globalThis.QOmniLib || {}));

  // ---- tensors from a QPACK1 container (f16 / f32 / i64 kinds; N-Q dense checkpoints have no tern5 / bits)
  function f16ToF32Array(bytes) {
    const n = bytes.length >> 1, out = new Float32Array(n), u = new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
    const tab = new Float32Array(65536);       // full fp16 decode table: exact, and fast for 46M weights
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      tab[h] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    for (let i = 0; i < n; i++) out[i] = tab[u[i]];
    return out;
  }

  function loadNQ(c) {
    const man = c.json('manifest.json'), cfg = c.json('config.json'), t = {};
    for (const e of man.tensors) {
      const b = c.blob(e.blob), copy = b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
      let data;
      if (e.kind === 'f16') data = f16ToF32Array(b);
      else if (e.kind === 'f32') data = new Float32Array(copy);
      else if (e.kind === 'i64') data = new BigInt64Array(copy);
      else if (e.kind === 'q8') {   // QPACK2: int8 rows + float32 per-row scale
        const sb = c.blob(e.scale);
        t[e.name] = { q8: new Int8Array(copy), scale: new Float32Array(sb.buffer.slice(sb.byteOffset, sb.byteOffset + sb.length)), shape: e.shape };
        continue;
      }
      else throw new Error(`tensor kind ${e.kind} (${e.name}) is not supported by the N-Q runtime`);
      t[e.name] = { data, shape: e.shape };
    }
    if (cfg.ternary || cfg.fingerprint || cfg.num_experts) throw new Error('this runtime implements the dense N-Q checkpoints only');
    return { cfg, tensors: t, selftest: c.toc['selftest.json'] ? c.json('selftest.json') : null };
  }

  // ---- math helpers (float64 accumulation, float32 storage)
  function matvec(W, x, rows, cols, out) {      // out[r] = sum_c W[r*cols+c] * x[c]
    for (let r = 0; r < rows; r++) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, base = r * cols, c = 0;
      for (; c + 3 < cols; c += 4) { s0 += W[base + c] * x[c]; s1 += W[base + c + 1] * x[c + 1]; s2 += W[base + c + 2] * x[c + 2]; s3 += W[base + c + 3] * x[c + 3]; }
      for (; c < cols; c++) s0 += W[base + c] * x[c];
      out[r] = s0 + s1 + s2 + s3;
    }
    return out;
  }
  function matvecQ8(Q, scale, x, rows, cols, out) {   // int8 weights with a per-row scale (QPACK2 'q8')
    for (let r = 0; r < rows; r++) {
      let s = 0; const base = r * cols;
      for (let c = 0; c < cols; c++) s += Q[base + c] * x[c];
      out[r] = s * scale[r];
    }
    return out;
  }
  const sigmoid = (v) => 1 / (1 + Math.exp(-v));
  const silu = (v) => v / (1 + Math.exp(-v));
  const softplus = (v) => (v > 20 ? v : Math.log1p(Math.exp(v)));
  function rmsnormZC(x, w, eps, group, out) {   // zero-centred RMSNorm: y = x / rms(x) * (1 + w), per group of `group` elements
    for (let g0 = 0; g0 < x.length; g0 += group) {
      let ss = 0; for (let i = g0; i < g0 + group; i++) ss += x[i] * x[i];
      const inv = 1 / Math.sqrt(ss / group + eps);
      for (let i = g0; i < g0 + group; i++) out[i] = Math.fround(x[i] * inv) * (1 + w[i - g0 + (w.length === x.length ? g0 : 0)]);
    }
    return out;
  }

  class NQ4JS {
    constructor(loaded) {
      const c = loaded.cfg; this.cfg = c; this.t = loaded.tensors;
      this.d = c.hidden_size; this.hc = c.hc_count; this.L = c.num_hidden_layers; this.eps = c.rms_norm_eps;
      this.V = c.vocab_size; this.rank = c.hc_lowrank; this.ff = c.intermediate_size;
      this.types = []; for (let i = 0; i < this.L; i++) this.types.push(c.layer_pattern[i % c.layer_pattern.length] === 'A' ? 'attn' : 'gdn');
      this.nk = c.linear_num_key_heads; this.nv = c.linear_num_value_heads; this.dk = c.linear_key_head_dim; this.dv = c.linear_value_head_dim;
      this.kdim = this.nk * this.dk; this.vdim = this.nv * this.dv; this.convC = 2 * this.kdim + this.vdim; this.K = c.linear_conv_kernel;
      this.nh = c.num_attention_heads; this.nkv = c.num_key_value_heads; this.hd = c.head_dim;
      this.rot = Math.floor(this.hd * c.partial_rotary_factor);
      this.inv = new Float64Array(this.rot / 2); for (let i = 0; i < this.rot / 2; i++) this.inv[i] = 1 / Math.pow(c.rope_theta, (2 * i) / this.rot);
      this.pleLayer = c.ple_layer ? c.ple_layer - 1 : -1;
      const e = c.eos_token_id; this.docEos = Array.isArray(e) ? e[e.length - 1] : e;
      if (this.pleLayer >= 0) {
        const P = `layers.${this.pleLayer}.ple.`;
        this.pSizes = Array.from(this.t[P + 'sizes'].data); this.pOff = Array.from(this.t[P + 'offsets'].data, Number);
        this.pMult = Array.from(this.t[P + 'mult'].data); this.pHeads = this.pSizes.length; this.pHpn = c.heads_per_ngram;
        this.pHdim = this.t[P + 'table.weight'].shape[1]; this.pDil = c.ngram_size; this.pK = c.ple_conv_kernel;
      }
      // scratch buffers
      const D = this.d, HD = this.hc * D;
      this.buf = { hn: new Float32Array(HD), low: new Float32Array(this.rank), gate: new Float32Array(HD), mixed: new Float32Array(D),
        inj: new Float32Array(this.hc), y: new Float32Array(D), gu: new Float32Array(2 * this.ff), act: new Float32Array(this.ff),
        qkvz: new Float32Array(this.convC + this.vdim), ba: new Float32Array(2 * this.nv), o: new Float32Array(this.vdim),
        q: new Float32Array(2 * this.nh * this.hd), k: new Float32Array(this.nkv * this.hd), v: new Float32Array(this.nkv * this.hd),
        ao: new Float32Array(this.nh * this.hd), logits: new Float32Array(this.V), emb: new Float32Array(D),
        trow: new Float32Array(this.pleLayer >= 0 ? this.pHdim : 1) };
      this.reset();
    }

    reset() {
      this.pos = 0; this.state = [];
      for (let l = 0; l < this.L; l++) {
        if (this.types[l] === 'gdn') this.state.push({ conv: [], S: Array.from({ length: this.nv }, () => new Float64Array(this.dk * this.dv)) });
        else this.state.push({ K: [], V: [] });
      }
      this.pleHist = [this.docEos, this.docEos]; this.pleConv = [];
    }

    // snapshot / restore of the whole decode state (cheap: lets the UI regenerate a reply without re-reading the chat)
    snapshot() { return { pos: this.pos, pleHist: this.pleHist.slice(), pleConv: this.pleConv.map((a) => a.slice()),
      state: this.state.map((s) => s.S ? { conv: s.conv.map((a) => a.slice()), S: s.S.map((m) => m.slice()) } : { K: s.K.slice(), V: s.V.slice() }) }; }
    restore(sn) { this.pos = sn.pos; this.pleHist = sn.pleHist.slice(); this.pleConv = sn.pleConv.map((a) => a.slice());
      this.state = sn.state.map((s) => s.S ? { conv: s.conv.map((a) => a.slice()), S: s.S.map((m) => m.slice()) } : { K: s.K.slice(), V: s.V.slice() }); }

    W(name) { const x = this.t[name]; if (!x) throw new Error('missing tensor ' + name); return x.data; }

    // y = W x for the matrix `name` (rows x cols): WASM SIMD kernel when attached, plain JS otherwise; f32 or q8 weights
    mv(name, x, rows, cols, out) {
      const t = this.t[name]; if (!t) throw new Error('missing tensor ' + name);
      const k = this.kernel;
      if (t.ptr) {
        k.F32.set(x.length === cols ? x : x.subarray(0, cols), k.xPtr >> 2);
        if (t.q8) k.ex.matvec_q8(t.ptr, t.sptr, k.xPtr, k.yPtr, rows, cols); else k.ex.matvec_f32(t.ptr, k.xPtr, k.yPtr, rows, cols);
        out.set(k.F32.subarray(k.yPtr >> 2, (k.yPtr >> 2) + rows));
        return out;
      }
      return t.q8 ? matvecQ8(t.q8, t.scale, x, rows, cols, out) : matvec(t.data, x, rows, cols, out);
    }

    // one row of a table (embedding, n-gram table) as floats; `out` is reused
    row(name, i, out) {
      const t = this.t[name], cols = t.shape[1], k = this.kernel;
      if (t.ptr && t.q8) { k.ex.row_q8(t.ptr, t.sptr, i, cols, k.yPtr); out.set(k.F32.subarray(k.yPtr >> 2, (k.yPtr >> 2) + cols)); return out; }
      if (t.ptr) { out.set(k.F32.subarray((t.ptr >> 2) + i * cols, (t.ptr >> 2) + (i + 1) * cols)); return out; }
      if (t.q8) { const sc = t.scale[i]; for (let c = 0; c < cols; c++) out[c] = t.q8[i * cols + c] * sc; return out; }
      out.set(t.data.subarray(i * cols, (i + 1) * cols)); return out;
    }

    // the whole decode step in WebAssembly (nq_core.c): every tensor goes into linear memory, step()/reset()/snapshot()
    // become calls into the core. Same results as the JS path (float32 everywhere, like the PyTorch reference).
    attachCore(instance, maxCtx = 4096) {
      const ex = instance.exports, c = this.cfg, t = this.t;
      const LS = ['gr_mix.norm.weight', 'gr_mix.down.weight', 'gr_mix.up.weight', 'gr_mix.inject.weight', 'gr_ffn.norm.weight', 'gr_ffn.down.weight',
        'gr_ffn.up.weight', 'gr_ffn.inject.weight', 'ffn.gate_up.weight', 'ffn.down.weight', 'mixer.in_qkvz.weight', 'mixer.in_ba.weight', 'mixer.conv.weight',
        'mixer.A_log', 'mixer.dt_bias', 'mixer.norm.weight', 'mixer.out.weight', 'mixer.q.weight', 'mixer.k.weight', 'mixer.v.weight', 'mixer.o.weight',
        'mixer.q_norm.weight', 'mixer.k_norm.weight', 'ple.table.weight', 'ple.key.weight', 'ple.value.weight', 'ple.norm_k.weight', 'ple.norm_q.weight',
        'ple.norm_c.weight', 'ple.conv.weight', 'ple.sizes', 'ple.offsets', 'ple.mult'];
      const put = (slot, name) => {
        const x = t[name]; if (!x) return;
        const rows = x.shape[0], cols = x.shape.slice(1).reduce((a, b) => a * b, 1);
        if (x.q8) { const p = ex.alloc(x.q8.length), sp = ex.alloc(4 * rows); new Int8Array(ex.memory.buffer, p, x.q8.length).set(x.q8); new Float32Array(ex.memory.buffer, sp, rows).set(x.scale); ex.nq_set(slot, p, sp, rows, cols); }
        else if (x.data instanceof BigInt64Array) { const p = ex.alloc(8 * x.data.length); new BigInt64Array(ex.memory.buffer, p, x.data.length).set(x.data); ex.nq_set(slot, p, 0, rows, cols); }
        else { const p = ex.alloc(4 * x.data.length); new Float32Array(ex.memory.buffer, p, x.data.length).set(x.data); ex.nq_set(slot, p, 0, rows, cols); }
      };
      put(0, 'embed.weight'); put(1, 'lm_head.weight'); put(2, 'mixer.norm.weight'); put(3, 'mixer.down.weight'); put(4, 'mixer.up.weight');
      for (let l = 0; l < this.L; l++) LS.forEach((n, id) => put(8 + l * 40 + id, `layers.${l}.${n}`));
      const ple = this.pleLayer >= 0;
      const ints = [this.d, this.hc, this.L, this.rank, this.ff, this.V, this.nk, this.nv, this.dk, this.dv, this.K, this.nh, this.nkv, this.hd, this.rot,
        this.pleLayer, ple ? this.pHeads : 0, ple ? this.pHpn : 0, ple ? this.pHdim : 0, ple ? this.pDil : 1, ple ? this.pK : 1, this.docEos, maxCtx];
      const cp = ex.nq_cfg(), I = new Int32Array(ex.memory.buffer, cp, 64), F = new Float32Array(ex.memory.buffer, cp, 64);
      ints.forEach((v, i) => { I[i] = v; }); F[23] = c.rms_norm_eps; F[24] = c.rope_theta;
      this.types.forEach((ty, l) => { I[25 + l] = ty === 'attn' ? 1 : 0; });
      if (!ex.nq_init()) throw new Error('out of WASM memory');
      new Float32Array(ex.memory.buffer, ex.nq_inv(), this.inv.length).set(this.inv);   // RoPE inverse frequencies (no pow() without libc)
      this.core = { ex, maxCtx, statePtr: ex.nq_state_ptr(), stateBytes: Number(ex.nq_state_bytes()) };
      for (const k of Object.keys(t)) if (!k.startsWith('decide_head')) { t[k].data = null; t[k].q8 = null; t[k].scale = null; }   // released: the core owns them
      const mem = () => ex.memory.buffer;
      this.step = (tok, want = true) => {
        const ptr = ex.nq_step(tok, want === 'hidden' ? 2 : want ? 1 : 0);
        if (!want) return null;
        return want === 'hidden' ? new Float32Array(mem(), ptr, this.d).slice() : new Float32Array(mem(), ptr, this.V);   // logits: a view, valid until the next step
      };
      this.reset = () => ex.nq_reset();
      this.snapshot = () => new Uint8Array(mem(), this.core.statePtr, this.core.stateBytes).slice();
      this.restore = (sn) => new Uint8Array(mem(), this.core.statePtr, this.core.stateBytes).set(sn);
      Object.defineProperty(this, 'pos', { get: () => ex.nq_pos(), configurable: true });
      ex.nq_reset();
      return this.core.stateBytes;
    }

    // move every sizeable matrix into WASM linear memory (once); afterwards mv()/row() run in the SIMD kernel
    attachKernel(instance) {
      const ex = instance.exports, big = Object.entries(this.t).filter(([, t]) => t.shape.length === 2 && t.shape[0] * t.shape[1] >= 4096 && (t.data instanceof Float32Array || t.q8));
      const k = { ex, xPtr: ex.alloc(4 * 4096), yPtr: ex.alloc(4 * Math.max(this.V, 4096)) };
      for (const [, t] of big) {
        const n = t.shape[0] * t.shape[1];
        if (t.q8) { t.ptr = ex.alloc(n); t.sptr = ex.alloc(4 * t.shape[0]); if (!t.ptr || !t.sptr) throw new Error('out of WASM memory'); }
        else { t.ptr = ex.alloc(4 * n); if (!t.ptr) throw new Error('out of WASM memory'); }
      }
      const mem = ex.memory.buffer; k.F32 = new Float32Array(mem);      // the memory has stopped growing: one view for all
      for (const [, t] of big) {
        if (t.q8) { new Int8Array(mem, t.ptr, t.q8.length).set(t.q8); k.F32.set(t.scale, t.sptr >> 2); t.q8 = true; t.scale = null; }
        else { k.F32.set(t.data, t.ptr >> 2); t.data = null; }           // the JS copy is released
      }
      this.kernel = k; return big.length;
    }

    gatedResidual(P, h, withInject) {      // -> mixed (d) [, inject weights (hc)]
      const D = this.d, HC = this.hc, b = this.buf;
      rmsnormZC(h, this.W(P + 'norm.weight'), this.eps, D, b.hn);
      this.mv(P + 'down.weight', b.hn, this.rank, HC * D, b.low);
      for (let i = 0; i < this.rank; i++) b.low[i] = silu(b.low[i] / HC);
      this.mv(P + 'up.weight', b.low, HC * D, this.rank, b.gate);
      for (let j = 0; j < D; j++) { let s = 0; for (let c = 0; c < HC; c++) s += sigmoid(b.gate[c * D + j]) * b.hn[c * D + j]; b.mixed[j] = s / HC; }
      if (withInject) { this.mv(P + 'inject.weight', b.hn, HC, HC * D, b.inj); for (let c = 0; c < HC; c++) b.inj[c] = 2 * sigmoid(b.inj[c] / HC); }
      return b.mixed;
    }

    write(h, y, w) { const D = this.d; for (let c = 0; c < this.hc; c++) for (let j = 0; j < D; j++) h[c * D + j] += y[j] * w[c]; }

    ngramIds(tok) {                          // the 2-gram and 3-gram ending at `tok`; never reaches across <eos> (document boundary)
      const eos = this.docEos, [p2, p1] = this.pleHist;
      const s1 = p1, s2 = p1 === eos ? eos : p2;       // see PLE.ngram_ids: a shifted token is replaced by <eos> once an <eos> lies between
      const m = this.pMult, sh = [BigInt(tok), BigInt(s1), BigInt(s2)], ids = [];
      for (let ng = 2; ng <= 3; ng++) {
        let hsh = sh[0] * m[0];
        for (let s = 1; s < ng; s++) hsh ^= sh[s] * m[s];      // BigInt: products reach 2^54, beyond float64 integer precision
        for (let k = (ng - 2) * this.pHpn; k < (ng - 1) * this.pHpn; k++) ids.push(Number(hsh % this.pSizes[k]) + this.pOff[k]);
      }
      return ids;
    }

    ple(h, tok) {                             // returns the PLE term (hc*d) added to the residual streams
      const P = `layers.${this.pleLayer}.ple.`, D = this.d, HC = this.hc, HD = HC * D;
      const ids = this.ngramIds(tok); this.pleHist = [this.pleHist[1], tok];
      const hdim = this.pHdim, e = new Float32Array(ids.length * hdim);
      ids.forEach((id, j) => e.set(this.row(P + 'table.weight', id, this.buf.trow), j * hdim));
      const k = rmsnormZC(this.mv(P + 'key.weight', e, HD, e.length, new Float32Array(HD)), this.W(P + 'norm_k.weight'), this.eps, D, new Float32Array(HD));
      const v = this.mv(P + 'value.weight', e, D, e.length, new Float32Array(D));
      const qn = rmsnormZC(h, this.W(P + 'norm_q.weight'), this.eps, D, new Float32Array(HD));
      const gv = new Float32Array(HD);
      for (let c = 0; c < HC; c++) {
        let s = 0; for (let j = 0; j < D; j++) s += k[c * D + j] * qn[c * D + j];
        s /= Math.sqrt(D); s = Math.sqrt(Math.max(Math.abs(s), 1e-6)) * Math.sign(s);
        const g = sigmoid(s); for (let j = 0; j < D; j++) gv[c * D + j] = g * v[j];
      }
      const cn = rmsnormZC(gv, this.W(P + 'norm_c.weight'), this.eps, D, new Float32Array(HD));
      const span = (this.pK - 1) * this.pDil, cw = this.W(P + 'conv.weight');
      this.pleConv.push(cn); if (this.pleConv.length > span + 1) this.pleConv.shift();
      const out = new Float32Array(HD), n = this.pleConv.length;
      for (let ch = 0; ch < HD; ch++) {
        let s = 0;
        for (let j = 0; j < this.pK; j++) { const back = (this.pK - 1 - j) * this.pDil, idx = n - 1 - back; if (idx >= 0) s += cw[ch * this.pK + j] * this.pleConv[idx][ch]; }
        out[ch] = gv[ch] + silu(s);
      }
      return out;
    }

    gdn(l, x) {
      const P = `layers.${l}.mixer.`, st = this.state[l], b = this.buf, C = this.convC, K = this.K;
      this.mv(P + 'in_qkvz.weight', x, C + this.vdim, this.d, b.qkvz);
      this.mv(P + 'in_ba.weight', x, 2 * this.nv, this.d, b.ba);
      st.conv.push(b.qkvz.slice(0, C)); if (st.conv.length > K) st.conv.shift();
      const cw = this.W(P + 'conv.weight'), n = st.conv.length, xc = new Float32Array(C);
      for (let ch = 0; ch < C; ch++) { let s = 0; for (let j = 0; j < K; j++) { const idx = n - K + j; if (idx >= 0) s += cw[ch * K + j] * st.conv[idx][ch]; } xc[ch] = silu(s); }
      const dk = this.dk, dv = this.dv, rep = this.nv / this.nk, scale = 1 / Math.sqrt(dk);
      const qh = [], kh = [];
      for (let hI = 0; hI < this.nk; hI++) {    // l2-normalise q, k per key head (F.normalize, eps 1e-6)
        let sq = 0, sk = 0; for (let i = 0; i < dk; i++) { sq += xc[hI * dk + i] ** 2; sk += xc[this.kdim + hI * dk + i] ** 2; }
        const nq = Math.max(Math.sqrt(sq), 1e-6), nk = Math.max(Math.sqrt(sk), 1e-6), q = new Float64Array(dk), k = new Float64Array(dk);
        for (let i = 0; i < dk; i++) { q[i] = xc[hI * dk + i] / nq * scale; k[i] = xc[this.kdim + hI * dk + i] / nk; }
        qh.push(q); kh.push(k);
      }
      const A = this.W(P + 'A_log'), dtb = this.W(P + 'dt_bias'), nw = this.W(P + 'norm.weight');
      for (let vh = 0; vh < this.nv; vh++) {
        const q = qh[(vh / rep) | 0], k = kh[(vh / rep) | 0], S = st.S[vh];
        const beta = sigmoid(b.ba[vh]), g = Math.exp(-Math.exp(A[vh]) * softplus(b.ba[this.nv + vh] + dtb[vh]));
        const vOff = 2 * this.kdim + vh * dv;
        for (let i = 0; i < dk * dv; i++) S[i] *= g;
        const kS = new Float64Array(dv);
        for (let i = 0; i < dk; i++) { const ki = k[i], row = i * dv; for (let j = 0; j < dv; j++) kS[j] += ki * S[row + j]; }
        for (let j = 0; j < dv; j++) kS[j] = (xc[vOff + j] - kS[j]) * beta;
        for (let i = 0; i < dk; i++) { const ki = k[i], row = i * dv; for (let j = 0; j < dv; j++) S[row + j] += ki * kS[j]; }
        const o = new Float64Array(dv);
        for (let i = 0; i < dk; i++) { const qi = q[i], row = i * dv; for (let j = 0; j < dv; j++) o[j] += qi * S[row + j]; }
        let ss = 0; for (let j = 0; j < dv; j++) ss += o[j] * o[j];
        const inv = 1 / Math.sqrt(ss / dv + this.eps);
        for (let j = 0; j < dv; j++) b.o[vh * dv + j] = Math.fround(o[j] * inv) * (1 + nw[j]) * sigmoid(b.qkvz[C + vh * dv + j]);
      }
      return this.mv(P + 'out.weight', b.o, this.d, this.vdim, b.y);
    }

    attn(l, x) {
      const P = `layers.${l}.mixer.`, st = this.state[l], b = this.buf, hd = this.hd, nh = this.nh, nkv = this.nkv;
      this.mv(P + 'q.weight', x, 2 * nh * hd, this.d, b.q);
      this.mv(P + 'k.weight', x, nkv * hd, this.d, b.k);
      this.mv(P + 'v.weight', x, nkv * hd, this.d, b.v);
      const qn = this.W(P + 'q_norm.weight'), kn = this.W(P + 'k_norm.weight'), r = this.rot, half = r / 2;
      const cos = new Float64Array(half), sin = new Float64Array(half);
      for (let i = 0; i < half; i++) { const a = Math.fround(this.pos * this.inv[i]); cos[i] = Math.fround(Math.cos(a)); sin[i] = Math.fround(Math.sin(a)); }
      const ropeHead = (src, off, w) => {       // zero-centred RMSNorm over the head, then partial RoPE on the first `r` dims
        const v = new Float32Array(hd); let ss = 0;
        for (let i = 0; i < hd; i++) ss += src[off + i] ** 2;
        const inv = 1 / Math.sqrt(ss / hd + this.eps);
        for (let i = 0; i < hd; i++) v[i] = Math.fround(src[off + i] * inv) * (1 + w[i]);
        const out = v.slice();
        for (let i = 0; i < half; i++) { out[i] = v[i] * cos[i] - v[i + half] * sin[i]; out[i + half] = v[i + half] * cos[i] + v[i] * sin[i]; }
        return out;
      };
      const ks = [], vs = [];
      for (let g = 0; g < nkv; g++) { ks.push(ropeHead(b.k, g * hd, kn)); vs.push(b.v.slice(g * hd, (g + 1) * hd)); }
      st.K.push(ks); st.V.push(vs);
      const T = st.K.length, rep = nh / nkv, sc = 1 / Math.sqrt(hd), scores = new Float64Array(T);
      for (let hI = 0; hI < nh; hI++) {
        const q = ropeHead(b.q, hI * 2 * hd, qn), g = (hI / rep) | 0;      // per head: [query (hd) | gate (hd)]
        let mx = -Infinity;
        for (let t = 0; t < T; t++) { const kk = st.K[t][g]; let s = 0; for (let i = 0; i < hd; i++) s += q[i] * kk[i]; s *= sc; scores[t] = s; if (s > mx) mx = s; }
        let sum = 0; for (let t = 0; t < T; t++) { scores[t] = Math.exp(scores[t] - mx); sum += scores[t]; }
        for (let i = 0; i < hd; i++) {
          let o = 0; for (let t = 0; t < T; t++) o += scores[t] * st.V[t][g][i];
          b.ao[hI * hd + i] = (o / sum) * sigmoid(b.q[hI * 2 * hd + hd + i]);
        }
      }
      return this.mv(P + 'o.weight', b.ao, this.d, nh * hd, b.y);
    }

    ffn(l, x) {
      const P = `layers.${l}.ffn.`, b = this.buf, ff = this.ff;
      this.mv(P + 'gate_up.weight', x, 2 * ff, this.d, b.gu);
      for (let i = 0; i < ff; i++) b.act[i] = silu(b.gu[i]) * b.gu[ff + i];
      return this.mv(P + 'down.weight', b.act, this.d, ff, b.y);
    }

    // one decode step: token id -> logits (Float32Array, reused between calls: copy it if you keep it).
    // wantLogits=false skips the 8256x512 output matmul (prompt prefill: only the state matters);
    // wantLogits='hidden' returns the final hidden state (d floats, a copy) instead — what the JEV decision head reads.
    step(tok, wantLogits = true) {
      const D = this.d, HC = this.hc, emb = this.row('embed.weight', tok, this.buf.emb), h = new Float32Array(HC * D);
      for (let c = 0; c < HC; c++) h.set(emb, c * D);
      for (let l = 0; l < this.L; l++) {
        if (l === this.pleLayer) { const p = this.ple(h, tok); for (let i = 0; i < h.length; i++) h[i] += p[i]; }
        let x = this.gatedResidual(`layers.${l}.gr_mix.`, h, true).slice(); let w = this.buf.inj.slice();
        this.write(h, this.types[l] === 'gdn' ? this.gdn(l, x) : this.attn(l, x), w);
        x = this.gatedResidual(`layers.${l}.gr_ffn.`, h, true).slice(); w = this.buf.inj.slice();
        this.write(h, this.ffn(l, x), w);
      }
      if (this.pleLayer < 0) this.pleHist = [this.pleHist[1], tok];
      this.pos += 1;
      if (!wantLogits) return null;
      const fin = this.gatedResidual('mixer.', h, false);
      if (wantLogits === 'hidden') return fin.slice();
      return this.mv('lm_head.weight', fin, this.V, D, this.buf.logits);
    }

    // JEV decision head on a final hidden state (NQ4ForDecision.decide_head, clamped like option_logits)
    decisionScore(hidden) {
      const w = this.W('decide_head.weight'), b = this.W('decide_head.bias');
      let s = b[0]; for (let i = 0; i < hidden.length; i++) s += w[i] * hidden[i];
      return Math.max(-30, Math.min(30, s));
    }
  }

  Object.assign(Lib, { loadNQ, NQ4JS });
  if (typeof module !== 'undefined') module.exports = { loadNQ, NQ4JS };
})();
