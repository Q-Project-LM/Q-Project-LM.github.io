// Host wrapper for kernel.wasm: uploads the model's tensors into the module's linear memory once, then exposes the
// same synchronous step(id) interface as model.js's QOmniJS — but each call runs the whole forward pass as a single
// compiled, SIMD-accelerated WebAssembly call instead of interpreted JavaScript.
'use strict';

async function loadWasmModel(loaded, wasmBytes) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  const write = (ptr, arr) => new Float32Array(ex.memory.buffer, ptr, arr.length).set(arr);
  const t = loaded.tensors;
  write(ex.get_table(), loaded.table);
  write(ex.get_logit_bias(), t['logit_bias'].data);
  write(ex.get_w_in(), t['model.w_in.weight'].data);
  write(ex.get_w_out(), t['w_out.weight'].data);
  write(ex.get_final_norm(), t['model.norm.weight'].data);
  const L = loaded.cfg.num_hidden_layers;
  for (let l = 0; l < L; l++) {
    const P = `model.layers.${l}.`;
    write(ex.get_n1(l), t[P + 'n1.weight'].data);
    write(ex.get_n2(l), t[P + 'n2.weight'].data);
    write(ex.get_qn(l), t[P + 'attn.q_norm.weight'].data);
    write(ex.get_kn(l), t[P + 'attn.k_norm.weight'].data);
    write(ex.get_gate_b(l), t[P + 'attn.out_gate.bias'].data);
    write(ex.get_q(l), t[P + 'attn.q_proj.weight'].data);
    write(ex.get_k(l), t[P + 'attn.k_proj.weight'].data);
    write(ex.get_v(l), t[P + 'attn.v_proj.weight'].data);
    write(ex.get_o(l), t[P + 'attn.o_proj.weight'].data);
    write(ex.get_gate(l), t[P + 'attn.out_gate.weight'].data);
    write(ex.get_w1(l), t[P + 'mlp.w1.weight'].data);
    write(ex.get_w2(l), t[P + 'mlp.w2.weight'].data);
  }
  return new QOmniWasm(ex, loaded.cfg);
}

class QOmniWasm {
  constructor(ex, cfg) {
    this.ex = ex; this.cfg = cfg; this.hd = cfg.head_dim;
    const invFreq = new Float32Array(this.hd / 2);
    for (let i = 0; i < invFreq.length; i++) invFreq[i] = 1 / Math.pow(cfg.rope_theta, (2 * i) / this.hd);
    this.invFreq = invFreq;
    this.reset();
  }

  reset() { this.pos = 0; } // the KV cache lives in WASM memory; positions beyond the new pos are simply never read again

  step(id) {
    const ex = this.ex, hd = this.hd;
    const cos = new Float32Array(hd), sin = new Float32Array(hd);
    for (let i = 0; i < hd / 2; i++) { const a = this.pos * this.invFreq[i]; const cv = Math.cos(a), sv = Math.sin(a); cos[i] = cv; cos[i + hd / 2] = cv; sin[i] = sv; sin[i + hd / 2] = sv; }
    new Float32Array(ex.memory.buffer, ex.get_cos(), hd).set(cos);
    new Float32Array(ex.memory.buffer, ex.get_sin(), hd).set(sin);
    const ptr = ex.step(id, this.pos);
    this.pos += 1;
    return new Float32Array(ex.memory.buffer, ptr, this.cfg.vocab_size).slice(); // copy out: the WASM buffer is reused next call
  }
}

window.QOmniLib = window.QOmniLib || {};
Object.assign(window.QOmniLib, { loadWasmModel, QOmniWasm });
