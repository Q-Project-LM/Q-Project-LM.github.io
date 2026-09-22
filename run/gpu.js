// WebGPU-accelerated Q-Omni: identical algorithm and op order to model.js (QOmniJS), verified byte-exact against the
// reference PyTorch model on real weights (see README). Only the linear() (GEMV) calls run on the GPU; RMSNorm, RoPE,
// softmax attention and the tiny per-head gate stay on the CPU since they are cheap relative to the matmuls.
// Weight buffers are uploaded to the GPU once at load time; only the small activation vector round-trips per call.
'use strict';

const GEMV_WGSL = `
struct Params { outDim: u32, inDim: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = gid.x;
  if (o >= p.outDim) { return; }
  var s: f32 = 0.0;
  let base = o * p.inDim;
  for (var k: u32 = 0u; k < p.inDim; k = k + 1u) {
    s = s + W[base + k] * x[k];
  }
  y[o] = s;
}`;

class GPUGemv {
  // One instance per weight matrix W [outDim, inDim] (row-major, same layout as F.linear's weight): y = W @ x.
  constructor(ctx, W, outDim, inDim) {
    this.ctx = ctx; this.outDim = outDim; this.inDim = inDim;
    const dev = ctx.device;
    this.wBuf = dev.createBuffer({ size: W.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(this.wBuf, 0, W);
    this.xBuf = dev.createBuffer({ size: Math.max(4, inDim * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.yBuf = dev.createBuffer({ size: Math.max(4, outDim * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.staging = dev.createBuffer({ size: Math.max(4, outDim * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.params = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // WGSL uniform structs must be a multiple of 16 bytes
    dev.queue.writeBuffer(this.params, 0, new Uint32Array([outDim, inDim, 0, 0]));
    this.bindGroup = dev.createBindGroup({
      layout: ctx.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.wBuf } },
        { binding: 1, resource: { buffer: this.xBuf } },
        { binding: 2, resource: { buffer: this.yBuf } },
        { binding: 3, resource: { buffer: this.params } },
      ],
    });
  }

  async run(x) {
    const { device, pipeline } = this.ctx;
    device.queue.writeBuffer(this.xBuf, 0, x.buffer, x.byteOffset, x.byteLength);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.outDim / 64)); pass.end();
    enc.copyBufferToBuffer(this.yBuf, 0, this.staging, 0, this.outDim * 4);
    device.queue.submit([enc.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();
    return out;
  }
}

async function initGPU() {
  if (!navigator.gpu) throw new Error('WebGPU not available in this browser');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: GEMV_WGSL });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  return { device, pipeline };
}

// Same forward pass as model.js's QOmniJS, but every linear() is a GPUGemv.run() (async). Weight upload happens once
// in the constructor; per-token cost is just the small activation round trips.
class QOmniGPU {
  constructor(ctx, loaded) {
    this.ctx = ctx; this.cfg = loaded.cfg; this.table = loaded.table; const t = loaded.tensors;
    const c = this.cfg;
    this.nh = c.num_attention_heads; this.nkv = c.num_key_value_heads; this.hd = c.head_dim;
    this.d = c.hidden_size; this.ff = c.intermediate_size; this.L = c.num_hidden_layers;
    this.eps = c.rms_norm_eps; this.bits = c.code_bits;
    const g = (name, o, i) => new GPUGemv(ctx, t[name].data, o, i);
    this.g_win = g('model.w_in.weight', this.d, this.bits);
    this.g_wout = g('w_out.weight', this.bits, this.d);
    this.tableGemv = new GPUGemv(ctx, this.table, c.vocab_size, this.bits); // logits: table @ z
    this.logitBias = t['logit_bias'].data;
    this.finalNorm = t['model.norm.weight'].data;
    this.layers = [];
    for (let l = 0; l < this.L; l++) {
      const P = `model.layers.${l}.`;
      this.layers.push({
        n1: t[P + 'n1.weight'].data, n2: t[P + 'n2.weight'].data,
        qn: t[P + 'attn.q_norm.weight'].data, kn: t[P + 'attn.k_norm.weight'].data,
        gateB: t[P + 'attn.out_gate.bias'].data,
        q: g(P + 'attn.q_proj.weight', this.nh * this.hd, this.d),
        k: g(P + 'attn.k_proj.weight', this.nkv * this.hd, this.d),
        v: g(P + 'attn.v_proj.weight', this.nkv * this.hd, this.d),
        o: g(P + 'attn.o_proj.weight', this.d, this.nh * this.hd),
        gate: g(P + 'attn.out_gate.weight', this.nh, this.d),
        w1: g(P + 'mlp.w1.weight', this.ff, this.d),
        w2: g(P + 'mlp.w2.weight', this.d, this.ff),
      });
    }
    const invFreq = new Float32Array(this.hd / 2);
    for (let i = 0; i < invFreq.length; i++) invFreq[i] = 1 / Math.pow(c.rope_theta, (2 * i) / this.hd);
    this.invFreq = invFreq;
    this.reset();
  }

  reset() { this.kCache = []; this.vCache = []; for (let l = 0; l < this.L; l++) { this.kCache.push([]); this.vCache.push([]); } this.pos = 0; }

  ropeAt(pos) {
    const hd = this.hd, cos = new Float32Array(hd), sin = new Float32Array(hd);
    for (let i = 0; i < hd / 2; i++) { const a = pos * this.invFreq[i]; const cv = Math.cos(a), sv = Math.sin(a); cos[i] = cv; cos[i + hd / 2] = cv; sin[i] = sv; sin[i + hd / 2] = sv; }
    return { cos, sin };
  }

  applyRope(vec, cos, sin) {
    const hd = vec.length, half = hd / 2, out = new Float32Array(hd);
    for (let i = 0; i < half; i++) { out[i] = vec[i] * cos[i] - vec[i + half] * sin[i]; out[i + half] = vec[i + half] * cos[i + half] + vec[i] * sin[i + half]; }
    return out;
  }

  static rmsnorm(x, w, eps) {
    let ss = 0; for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const inv = 1 / Math.sqrt(ss / x.length + eps), y = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = x[i] * inv * w[i];
    return y;
  }

  async embed(id) {
    const bits = this.bits, row = this.table.subarray(id * bits, id * bits + bits);
    return this.g_win.run(row);
  }

  async logits(x) {
    const z = await this.g_wout.run(x);
    const raw = await this.tableGemv.run(z), scale = 1 / Math.sqrt(this.bits), out = new Float32Array(raw.length);
    for (let v = 0; v < raw.length; v++) out[v] = raw[v] * scale + this.logitBias[v];
    return out;
  }

  async step(id) {
    let x = await this.embed(id);
    const { cos, sin } = this.ropeAt(this.pos);
    const rep = this.nh / this.nkv, R = QOmniGPU.rmsnorm;
    for (let l = 0; l < this.L; l++) {
      const L = this.layers[l];
      const h = R(x, L.n1, this.eps);
      const [qFlat, kFlat, vFlat, gateLogit] = await Promise.all([L.q.run(h), L.k.run(h), L.v.run(h), L.gate.run(h)]);
      const qHeads = [], kHeads = [], vHeads = [];
      for (let hI = 0; hI < this.nh; hI++) qHeads.push(this.applyRope(R(qFlat.subarray(hI * this.hd, (hI + 1) * this.hd), L.qn, this.eps), cos, sin));
      for (let hI = 0; hI < this.nkv; hI++) { kHeads.push(this.applyRope(R(kFlat.subarray(hI * this.hd, (hI + 1) * this.hd), L.kn, this.eps), cos, sin)); vHeads.push(vFlat.slice(hI * this.hd, (hI + 1) * this.hd)); }
      for (let hI = 0; hI < this.nkv; hI++) { this.kCache[l][hI] = this.kCache[l][hI] || []; this.vCache[l][hI] = this.vCache[l][hI] || []; this.kCache[l][hI].push(kHeads[hI]); this.vCache[l][hI].push(vHeads[hI]); }
      const attnOut = new Float32Array(this.nh * this.hd);
      for (let hI = 0; hI < this.nh; hI++) {
        const kv = (hI / rep) | 0, kc = this.kCache[l][kv], vc = this.vCache[l][kv], T = kc.length;
        const scores = new Float32Array(T); const scale = 1 / Math.sqrt(this.hd); let mx = -Infinity;
        for (let ti = 0; ti < T; ti++) { let s = 0; for (let k = 0; k < this.hd; k++) s += qHeads[hI][k] * kc[ti][k]; s *= scale; scores[ti] = s; if (s > mx) mx = s; }
        let sum = 0; for (let ti = 0; ti < T; ti++) { scores[ti] = Math.exp(scores[ti] - mx); sum += scores[ti]; }
        const o = new Float32Array(this.hd);
        for (let ti = 0; ti < T; ti++) { const w = scores[ti] / sum; for (let k = 0; k < this.hd; k++) o[k] += w * vc[ti][k]; }
        const gate = 1 / (1 + Math.exp(-(gateLogit[hI] + L.gateB[hI])));
        for (let k = 0; k < this.hd; k++) attnOut[hI * this.hd + k] = o[k] * gate;
      }
      const o = await L.o.run(attnOut);
      for (let k = 0; k < this.d; k++) x[k] += o[k];
      const h2 = R(x, L.n2, this.eps);
      const u = await L.w1.run(h2);
      for (let k = 0; k < this.ff; k++) u[k] = u[k] / (1 + Math.exp(-u[k]));
      const mo = await L.w2.run(u);
      for (let k = 0; k < this.d; k++) x[k] += mo[k];
    }
    x = R(x, this.finalNorm, this.eps);
    this.pos += 1;
    return this.logits(x);
  }
}



window.QOmniLib = window.QOmniLib || {};
Object.assign(window.QOmniLib, { initGPU, QOmniGPU, GPUGemv });
