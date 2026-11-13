// gpt-oss (qpack container: q8g attention / head, q1 + optional second q1 plane and mxfp4 experts, f16 embedding) on WebGPU.
// One decode step = one command buffer: per layer rmsnorm -> q/k/v (q8g) -> bias + YaRN rope + KV write -> attention with sinks and
// the sliding window -> wo -> residual -> rmsnorm -> router (top-4 + softmax, on the GPU) -> the 4 experts' gate_up (one dispatch per
// expert kind; each workgroup reads the routed expert id and skips experts of other kinds) -> bias + clamped swiglu -> down -> weighted
// sum + residual. The embedding (1.16 GB f16) stays in JS memory and one row is uploaded per token, as the CUDA engine keeps it on the
// host. Semantics follow src/model_gptoss.cpp.
'use strict';
(function () {
  const G = typeof window !== 'undefined' ? window : globalThis;
  const Lib = (G.QPack = G.QPack || {});

  // ---------------------------------------------------------------- WGSL
  const COMMON = /* wgsl */`
fn f16at(s: ptr<storage, array<u32>, read>, i: u32) -> f32 { let w = (*s)[i >> 1u]; return unpack2x16float(w)[i & 1u]; }
fn u8at(s: ptr<storage, array<u32>, read>, i: u32) -> u32 { return ((*s)[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
fn i8of(w: u32, b: u32) -> f32 { return f32(bitcast<i32>(w << (24u - 8u * b)) >> 24u); }
`;
  // matvec: 256 threads = 4 rows x 64 lanes. mode 0: y[r] = W x; mode 1 (experts): workgroup_id.y = slot k, expert e = route[k],
  // m = map[e] (kind << 16 | local index | has_r2 << 24); slots whose expert is of another kind are skipped
  const MV = (kind) => /* wgsl */`
${COMMON}
struct P { rows: u32, cols: u32, wrow: u32, srow: u32, wexp: u32, sexp: u32, mode: u32, kind: u32, xslot: u32, r2: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> S: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> R: array<u32>;     // route: ids[4], weights[4] (bitcast)
@group(0) @binding(6) var<storage, read> M: array<u32>;     // expert map[32]
@group(0) @binding(7) var<storage, read> W2: array<u32>;    // second q1 plane (r2)
@group(0) @binding(8) var<storage, read> S2: array<u32>;
var<workgroup> red: array<f32, 256>;

fn rowdot(r: u32, lane: u32, wb: u32, sb: u32, xo: u32, second: bool) -> f32 {
  var acc = 0.0;
  ${kind === 'q8g' ? `
  let nw = p.cols / 4u;
  for (var w = lane; w < nw; w += 64u) {
    let word = W[wb + r * p.wrow + w]; let sc = f16at(&S, sb + r * p.srow + w / 8u); let c = xo + w * 4u;
    acc += sc * (i8of(word, 0u) * X[c] + i8of(word, 1u) * X[c + 1u] + i8of(word, 2u) * X[c + 2u] + i8of(word, 3u) * X[c + 3u]);
  }` : ''}
  ${kind === 'q1' ? `
  let nw = p.wrow;
  for (var w = lane; w < nw; w += 64u) {
    var word: u32; var sc: f32;
    if (second) { word = W2[wb + r * p.wrow + w]; sc = f16at(&S2, sb + r * p.srow + w / 4u); }
    else { word = W[wb + r * p.wrow + w]; sc = f16at(&S, sb + r * p.srow + w / 4u); }
    let c0 = w * 32u; let n = min(32u, p.cols - min(c0, p.cols)); var s = 0.0;
    for (var b = 0u; b < n; b++) { let xv = X[xo + c0 + b]; s += select(-xv, xv, ((word >> b) & 1u) == 1u); }
    acc += sc * s;
  }` : ''}
  ${kind === 'mx4' ? `
  let nw = p.cols / 8u;
  for (var w = lane; w < nw; w += 64u) {
    let word = W[wb + r * p.wrow + w]; let g = w / 4u; let j0 = (w % 4u) * 4u;
    let sc = exp2(f32(u8at(&S, sb + r * p.srow + g)) - 127.0); let c = xo + g * 32u + j0; var s = 0.0;
    for (var b = 0u; b < 4u; b++) {
      let byte = (word >> (8u * b)) & 255u;
      s += e2m1(byte & 15u) * X[c + b] + e2m1(byte >> 4u) * X[c + 16u + b];
    }
    acc += sc * s;
  }` : ''}
  ${kind === 'f32' ? `
  for (var c = lane; c < p.cols; c += 64u) { acc += bitcast<f32>(W[wb + r * p.wrow + c]) * X[xo + c]; }` : ''}
  return acc;
}
${kind === 'mx4' ? `fn e2m1(n: u32) -> f32 { let m = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0); return select(m[n & 7u], -m[n & 7u], (n & 8u) != 0u); }` : ''}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = (wg.x + wg.z * 65535u) * 4u + li / 64u; let lane = li % 64u;
  var wb = 0u; var sb = 0u; var xo = 0u; var yo = 0u; var two = false;
  if (p.mode == 1u) {
    let k = wg.y; let e = R[k]; let m = M[e];
    if (((m >> 16u) & 255u) != p.kind) { return; }
    let local = m & 65535u; wb = local * p.wexp; sb = local * p.sexp; xo = k * p.xslot; yo = k * p.rows;
    two = p.r2 == 1u && ((m >> 24u) & 1u) == 1u;
  }
  var v = 0.0;
  if (r < p.rows) { v = rowdot(r, lane, wb, sb, xo, false); if (two) { v += rowdot(r, lane, wb, sb, xo, true); } }
  red[li] = v; workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (lane < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  if (lane == 0u && r < p.rows) { Y[yo + r] = red[li]; }
}`;

  const SMALL = /* wgsl */`
struct St { pos: u32, lo_full: u32, lo_slide: u32, T: u32, eps: f32, limit: f32, alpha: f32, d: u32, cs: array<vec4<f32>, 8>, sn: array<vec4<f32>, 8> };
@group(0) @binding(0) var<uniform> st: St;
fn csv(j: u32) -> f32 { return st.cs[j / 4u][j % 4u]; }
fn snv(j: u32) -> f32 { return st.sn[j / 4u][j % 4u]; }
`;
  // rmsnorm: out = x / rms(x) * w   (one workgroup)
  const RMS = /* wgsl */`${SMALL}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wn: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li: u32) {
  var s = 0.0; for (var i = li; i < st.d; i += 256u) { s += X[i] * X[i]; }
  red[li] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(st.d) + st.eps);
  for (var i = li; i < st.d; i += 256u) { O[i] = X[i] * inv * Wn[i]; }
}`;
  // q/k biases, rope (rotate-half, concentration folded into cs/sn), K/V into the cache at pos
  const QKV = /* wgsl */`${SMALL}
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read> BQ: array<f32>;
@group(0) @binding(5) var<storage, read> BK: array<f32>;
@group(0) @binding(6) var<storage, read> BV: array<f32>;
@group(0) @binding(7) var<storage, read_write> KC: array<f32>;
@group(0) @binding(8) var<storage, read_write> VC: array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let h = wg.x;   // 0..63 query heads, 64..71 key heads (also write V)
  if (h < 64u) {
    let o = h * 64u; let a = Q[o + j] + BQ[o + j]; let b = Q[o + j + 32u] + BQ[o + j + 32u];
    Q[o + j] = a * csv(j) - b * snv(j); Q[o + j + 32u] = b * csv(j) + a * snv(j);
  } else {
    let g = h - 64u; let o = g * 64u; let a = K[o + j] + BK[o + j]; let b = K[o + j + 32u] + BK[o + j + 32u];
    let base = st.pos * 512u + o;
    KC[base + j] = a * csv(j) - b * snv(j); KC[base + j + 32u] = b * csv(j) + a * snv(j);
    VC[base + j] = V[o + j] + BV[o + j]; VC[base + j + 32u] = V[o + j + 32u] + BV[o + j + 32u];
  }
}`;
  // attention of the new token for one query head over [lo, T) + the learned sink logit
  const ATT = (MAXT) => /* wgsl */`${SMALL}
struct A { sliding: u32, p1: u32, p2: u32, p3: u32 };
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<f32>;
@group(0) @binding(3) var<storage, read> VC: array<f32>;
@group(0) @binding(4) var<storage, read> SK: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
@group(0) @binding(6) var<uniform> a: A;
var<workgroup> sc: array<f32, ${MAXT}>;
var<workgroup> red: array<f32, 64>;
var<workgroup> qs: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x; let g = h / 8u; let lo = select(st.lo_full, st.lo_slide, a.sliding == 1u); let n = st.T - lo;
  qs[li] = Q[h * 64u + li]; workgroupBarrier();
  var mx = SK[h];
  for (var t = li; t < n; t += 64u) {
    let kb = (lo + t) * 512u + g * 64u; var s = 0.0;
    for (var i = 0u; i < 64u; i++) { s += qs[i] * KC[kb + i]; }
    s *= 0.125; sc[t] = s; mx = max(mx, s);
  }
  red[li] = mx; workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (li < k) { red[li] = max(red[li], red[li + k]); } workgroupBarrier(); }
  let M = red[0]; workgroupBarrier();
  var sum = 0.0;
  for (var t = li; t < n; t += 64u) { let e = exp(sc[t] - M); sc[t] = e; sum += e; }
  red[li] = sum; workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let L = red[0] + exp(SK[h] - M);
  var acc = 0.0;
  for (var t = 0u; t < n; t++) { acc += sc[t] * VC[(lo + t) * 512u + g * 64u + li]; }
  O[h * 64u + li] = acc / L;
}`;
  // x += o + bias
  const RESID = /* wgsl */`${SMALL}
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> O: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i < st.d) { X[i] += O[i] + B[i]; } }`;
  // router: logits + bias, top-4 (descending, first index wins ties), softmax over the 4
  const ROUTE = /* wgsl */`${SMALL}
@group(0) @binding(1) var<storage, read> RL: array<f32>;
@group(0) @binding(2) var<storage, read> RB: array<f32>;
@group(0) @binding(3) var<storage, read_write> R: array<u32>;
@compute @workgroup_size(1) fn main() {
  var l: array<f32, 32>; for (var e = 0u; e < 32u; e++) { l[e] = RL[e] + RB[e] + 0.0 * st.eps; }
  var sel: array<u32, 4>; var v: array<f32, 4>;
  for (var k = 0u; k < 4u; k++) {
    var b = 99u;
    for (var e = 0u; e < 32u; e++) {
      var used = false; for (var j = 0u; j < k; j++) { if (sel[j] == e) { used = true; } }
      if (!used && (b == 99u || l[e] > l[b])) { b = e; }
    }
    sel[k] = b; v[k] = l[b];
  }
  var mx = v[0]; for (var k = 1u; k < 4u; k++) { mx = max(mx, v[k]); }
  var w: array<f32, 4>; var s = 0.0; for (var k = 0u; k < 4u; k++) { w[k] = exp(v[k] - mx); s += w[k]; }
  for (var k = 0u; k < 4u; k++) { R[k] = sel[k]; R[4u + k] = bitcast<u32>(w[k] / s); }
}`;
  // h[k][j] = glu(y[2j] + b) * (clamp(y[2j+1] + b) + 1)
  const SWIGLU = /* wgsl */`${SMALL}
@group(0) @binding(1) var<storage, read> Yg: array<f32>;
@group(0) @binding(2) var<storage, read> BG: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<u32>;
@group(0) @binding(4) var<storage, read_write> H: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.y; let j = gid.x; let ff = st.d; if (j >= ff) { return; }
  let e = R[k]; let y0 = Yg[k * 2u * ff + 2u * j] + BG[e * 2u * ff + 2u * j]; let y1 = Yg[k * 2u * ff + 2u * j + 1u] + BG[e * 2u * ff + 2u * j + 1u];
  let g = min(y0, st.limit); let u = clamp(y1, -st.limit, st.limit);
  H[k * ff + j] = g / (1.0 + exp(-st.alpha * g)) * (u + 1.0);
}`;
  // x += sum_k w_k (down_k + b_down[e_k])
  const COMBINE = /* wgsl */`${SMALL}
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> D: array<f32>;
@group(0) @binding(3) var<storage, read> BD: array<f32>;
@group(0) @binding(4) var<storage, read> R: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let d = st.d; if (i >= d) { return; }
  var s = 0.0; for (var k = 0u; k < 4u; k++) { let e = R[k]; s += bitcast<f32>(R[4u + k]) * (D[k * d + i] + BD[e * d + i]); }
  X[i] += s;
}`;

  // ---------------------------------------------------------------- helpers
  function f16tab() {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023; t[h] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024); }
    return t;
  }
  function parseToc(u8) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), n = dv.getUint32(8, true), toc = {};
    for (let i = 0; i < n; i++) {
      const b = 16 + 72 * i; let name = ''; for (let j = 0; j < 56 && u8[b + j]; j++) name += String.fromCharCode(u8[b + j]);
      toc[name] = [Number(dv.getBigUint64(b + 56, true)), Number(dv.getBigUint64(b + 64, true))];
    }
    return { n, toc, headEnd: 16 + 72 * n };
  }

  class GptOssWebGPU {
    constructor() { this.blobs = {}; this.json = {}; this.f16 = f16tab(); this.gpuBytes = 0; }

    static async device() {
      if (!G.navigator || !navigator.gpu) throw new Error('WebGPU is not available in this browser');
      const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!ad) throw new Error('no WebGPU adapter');
      const lim = ad.limits;
      const dev = await ad.requestDevice({ requiredLimits: { maxBufferSize: lim.maxBufferSize, maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxStorageBuffersPerShaderStage: Math.min(10, lim.maxStorageBuffersPerShaderStage) } });
      return { ad, dev, info: ad.info || {} };
    }

    // feed the container front to back: `chunk(u8)` in file order (from fetch or from a cached File); blobs are assembled and uploaded
    // as soon as they are complete. The embedding stays in JS, JSON blobs are parsed, everything else goes to a GPU buffer.
    begin(dev, onBlob) {
      this.dev = dev; this.pos = 0; this.head = new Uint8Array(0); this.order = null; this.cur = null; this.onBlob = onBlob;
    }
    feed(u8) {
      let off = 0;
      while (off < u8.length) {
        if (!this.order) {   // still reading the table of contents
          const need = this.head.length < 16 ? 16 : 16 + 72 * new DataView(this.head.buffer).getUint32(8, true);
          const take = Math.min(need - this.head.length, u8.length - off), h = new Uint8Array(this.head.length + take);
          h.set(this.head); h.set(u8.subarray(off, off + take), this.head.length); this.head = h; off += take; this.pos += take;
          if (this.head.length >= 16 && this.head.length === 16 + 72 * new DataView(this.head.buffer).getUint32(8, true)) {
            const t = parseToc(this.head); this.toc = t.toc; this.order = Object.entries(t.toc).sort((a, b) => a[1][0] - b[1][0]); this.oi = 0;
          }
          continue;
        }
        if (!this.cur) {
          if (this.oi >= this.order.length) return;
          const [name, [bo, bl]] = this.order[this.oi];
          if (this.pos < bo) { const skip = Math.min(bo - this.pos, u8.length - off); off += skip; this.pos += skip; continue; }
          this.cur = { name, len: bl, buf: new Uint8Array(bl), got: 0 };
        }
        const c = this.cur, take = Math.min(c.len - c.got, u8.length - off);
        c.buf.set(u8.subarray(off, off + take), c.got); c.got += take; off += take; this.pos += take;
        if (c.got === c.len) { this.cur = null; this.oi++; this.store(c.name, c.buf); }
      }
    }
    get done() { return this.order && this.oi >= this.order.length; }
    store(name, buf) {
      if (name.endsWith('.json')) { this.json[name] = buf; }
      else if (name === this.embedBlob()) { this.embed = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength >> 1); }
      else {
        const size = Math.max(4, (buf.byteLength + 3) & ~3), b = this.dev.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
        new Uint8Array(b.getMappedRange()).set(buf); b.unmap(); this.blobs[name] = b; this.gpuBytes += size;
      }
      if (this.onBlob) this.onBlob(name, this.pos);
    }
    embedBlob() {
      if (this._eb !== undefined) return this._eb;
      if (!this.json['manifest.json']) return null;
      const man = JSON.parse(new TextDecoder().decode(this.json['manifest.json']));
      this._eb = man.tensors.find((t) => t.name === 'embed').blob; return this._eb;
    }
    text(name) { return new TextDecoder().decode(this.json[name]); }

    // build pipelines, buffers and the per-layer bind groups
    init(maxCtx = 2048) {
      const dev = this.dev, cfg = this.cfg = JSON.parse(this.text('config.json')), man = JSON.parse(this.text('manifest.json'));
      const T = {}; for (const e of man.tensors) T[e.name] = e; this.T = T;
      const d = cfg.hidden_size, L = cfg.num_hidden_layers, E = cfg.num_local_experts, ff = cfg.intermediate_size, V = cfg.vocab_size;
      if (cfg.head_dim !== 64 || cfg.num_attention_heads !== 64 || cfg.num_key_value_heads !== 8 || cfg.num_experts_per_tok !== 4 || E > 32 || d !== ff)
        throw new Error('gpt-oss-20b geometry expected (64 heads, 8 KV heads, head_dim 64, top-4 of <= 32 experts)');
      Object.assign(this, { d, L, E, ff, V, ctx: maxCtx });
      const U = GPUBufferUsage;
      const buf = (bytes, usage = U.STORAGE | U.COPY_DST | U.COPY_SRC) => { this.gpuBytes += bytes; return dev.createBuffer({ size: Math.max(16, (bytes + 15) & ~15), usage }); };
      // explicit layouts (an 'auto' layout drops bindings a shader variant does not use): u = uniform, r = read-only storage, s = storage
      const pipe = (code, spec) => {
        const ty = { u: { buffer: { type: 'uniform' } }, r: { buffer: { type: 'read-only-storage' } }, s: { buffer: { type: 'storage' } } };
        const bgl = dev.createBindGroupLayout({ entries: [...spec].map((c, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, ...ty[c] })) });
        const p = dev.createComputePipeline({ layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' } });
        p.bgl = bgl; return p;
      };
      const MVS = 'rrrsurrrr';
      this.P = { q8g: pipe(MV('q8g'), MVS), q1: pipe(MV('q1'), MVS), mx4: pipe(MV('mx4'), MVS), f32: pipe(MV('f32'), MVS), rms: pipe(RMS, 'urrs'),
                 qkv: pipe(QKV, 'usrrrrrss'), att: pipe(ATT(maxCtx), 'urrrrsu'), resid: pipe(RESID, 'usrr'), route: pipe(ROUTE, 'urrs'),
                 swiglu: pipe(SWIGLU, 'urrrs'), combine: pipe(COMBINE, 'usrrr') };
      this.st = buf(32 + 256, U.UNIFORM | U.COPY_DST);
      const B = this.B = { x: buf(4 * d), xn: buf(4 * d), q: buf(4 * 4096), k: buf(4 * 512), v: buf(4 * 512), att: buf(4 * 4096), o: buf(4 * d), rl: buf(4 * 32),
                           route: buf(32), gu: buf(4 * 4 * 2 * ff), h: buf(4 * 4 * ff), dn: buf(4 * 4 * d), logits: buf(4 * V),
                           read: dev.createBuffer({ size: 4 * V, usage: U.MAP_READ | U.COPY_DST }), dummy: buf(16) };
      this.KC = []; this.VC = []; for (let l = 0; l < L; l++) { this.KC.push(buf(4 * maxCtx * 512)); this.VC.push(buf(4 * maxCtx * 512)); }
      // YaRN inverse frequencies and concentration
      const rs = cfg.rope_scaling || {}, theta = cfg.rope_theta || 150000, factor = rs.factor || 1, orig = rs.original_max_position_embeddings || 4096;
      const half = 32, inv = new Float64Array(half); let conc = 1;
      for (let i = 0; i < half; i++) inv[i] = 1 / Math.pow(theta, (2 * i) / 64);
      if (factor > 1) {
        conc = 0.1 * Math.log(factor) + 1;
        let low = half * Math.log(orig / ((rs.beta_fast || 32) * 2 * Math.PI)) / Math.log(theta), high = half * Math.log(orig / ((rs.beta_slow || 1) * 2 * Math.PI)) / Math.log(theta);
        if (rs.truncate) { low = Math.floor(low); high = Math.ceil(high); }
        for (let i = 0; i < half; i++) { const ex = inv[i], ip = inv[i] / factor, ramp = Math.min(1, Math.max(0, (i - low) / (high - low))), mask = 1 - ramp; inv[i] = ip * (1 - mask) + ex * mask; }
      }
      this.inv = inv; this.conc = conc;
      const lt = Array.isArray(cfg.layer_types) ? cfg.layer_types : null, mx = cfg.moe_exact || {}, r2sel = cfg.moe_r2 || null;
      const g = (name) => { const t = T[name]; if (!t) throw new Error('missing tensor ' + name); return t; };
      const blob = (t, key = 'blob') => { const b = this.blobs[t[key]]; if (!b) throw new Error('blob not loaded: ' + t[key]); return b; };
      const uni = (arr) => { const b = dev.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST }); dev.queue.writeBuffer(b, 0, new Uint32Array(arr.concat(new Array(12 - arr.length).fill(0)))); return b; };
      const geom = (t) => {   // words per row, scale elements per row
        const R = t.shape[t.shape.length - 2], C = t.shape[t.shape.length - 1];
        if (t.kind === 'q8g') return { R, C, wrow: C / 4, srow: C / 32 };
        if (t.kind === 'q1') { const Gq = Math.ceil(C / 128); return { R, C, wrow: Gq * 4, srow: Gq }; }
        if (t.kind === 'mxfp4') return { R, C, wrow: C / 8, srow: C / 32 };
        if (t.kind === 'f32') return { R, C, wrow: C, srow: 0 };
        throw new Error('unsupported kind ' + t.kind + ' for ' + t.name);
      };
      const KIND = { q1: 0, mxfp4: 1 };
      // a plain matvec (mode 0)
      const mv = (t, x, y) => {
        const gm = geom(t), k = t.kind === 'mxfp4' ? 'mx4' : t.kind;
        const bg = dev.createBindGroup({ layout: this.P[k].bgl, entries: [
          { binding: 0, resource: { buffer: blob(t) } }, { binding: 1, resource: { buffer: t.scale ? blob(t, 'scale') : B.dummy } },
          { binding: 2, resource: { buffer: x } }, { binding: 3, resource: { buffer: y } },
          { binding: 4, resource: { buffer: uni([gm.R, gm.C, gm.wrow, gm.srow, 0, 0, 0, 0, 0, 0]) } },
          { binding: 5, resource: { buffer: B.dummy } }, { binding: 6, resource: { buffer: B.dummy } }, { binding: 7, resource: { buffer: B.dummy } }, { binding: 8, resource: { buffer: B.dummy } }] });
        return { p: this.P[k], bg, n: Math.ceil(gm.R / 4), y: 1 };
      };
      // the experts of one kind for the 4 routed slots (mode 1)
      const mvx = (t, t2, map, x, xslot, y) => {
        const gm = geom(t), k = t.kind === 'mxfp4' ? 'mx4' : t.kind;
        const bg = dev.createBindGroup({ layout: this.P[k].bgl, entries: [
          { binding: 0, resource: { buffer: blob(t) } }, { binding: 1, resource: { buffer: blob(t, 'scale') } },
          { binding: 2, resource: { buffer: x } }, { binding: 3, resource: { buffer: y } },
          { binding: 4, resource: { buffer: uni([gm.R, gm.C, gm.wrow, gm.srow, gm.R * gm.wrow, gm.R * gm.srow, 1, KIND[t.kind], xslot, t2 ? 1 : 0]) } },
          { binding: 5, resource: { buffer: B.route } }, { binding: 6, resource: { buffer: map } },
          { binding: 7, resource: { buffer: t2 ? blob(t2) : B.dummy } }, { binding: 8, resource: { buffer: t2 ? blob(t2, 'scale') : B.dummy } }] });
        return { p: this.P[k], bg, n: Math.ceil(gm.R / 4), y: 4 };
      };
      const small = (p, bufs, n, y = 1, extra) => {
        const entries = [{ binding: 0, resource: { buffer: this.st } }].concat(bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } })));
        if (extra) entries.push(extra);
        return { p: this.P[p], bg: dev.createBindGroup({ layout: this.P[p].bgl, entries }), n, y };
      };
      this.steps = []; this.sliding = [];
      for (let l = 0; l < L; l++) {
        const n = (s) => g(`layers.${l}.${s}`), ls = String(l);
        const sliding = lt ? lt[l] === 'sliding_attention' : l % 2 === 0; this.sliding.push(sliding);
        const exact = mx[ls] || [], r2ids = r2sel && r2sel[ls] ? r2sel[ls] : null;
        const gu = n('w_gate_up'), dn = n('w_down'), gur = T[`layers.${l}.w_gate_up_r2`], dnr = T[`layers.${l}.w_down_r2`];
        const gux = T[`layers.${l}.w_gate_up_x`], dnx = T[`layers.${l}.w_down_x`];
        const mp = new Uint32Array(32); let iq = 0;
        for (let e = 0; e < E; e++) {
          const j = exact.indexOf(e);
          if (j >= 0) mp[e] = (KIND.mxfp4 << 16) | j;
          else {
            let hasR2 = gur ? 1 : 0;
            if (r2ids) throw new Error('moe_r2 (per-expert second planes) is not supported in the browser yet');
            mp[e] = (hasR2 << 24) | (KIND.q1 << 16) | iq; iq++;
          }
        }
        if (gu.kind !== 'q1' || (gux && gux.kind !== 'mxfp4')) throw new Error('expected q1 experts (+ mxfp4 exact ones)');
        const map = buf(128); dev.queue.writeBuffer(map, 0, mp);
        const att = dev.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }); dev.queue.writeBuffer(att, 0, new Uint32Array([sliding ? 1 : 0, 0, 0, 0]));
        const s = [];
        s.push(small('rms', [B.x, blob(n('attn_norm')), B.xn], 1));
        s.push(mv(n('wq'), B.xn, B.q)); s.push(mv(n('wk'), B.xn, B.k)); s.push(mv(n('wv'), B.xn, B.v));
        s.push(small('qkv', [B.q, B.k, B.v, blob(n('bq')), blob(n('bk')), blob(n('bv')), this.KC[l], this.VC[l]], 72));
        s.push(small('att', [B.q, this.KC[l], this.VC[l], blob(n('sinks')), B.att], 64, 1, { binding: 6, resource: { buffer: att } }));
        s.push(mv(n('wo'), B.att, B.o));
        s.push(small('resid', [B.x, B.o, blob(n('bo'))], Math.ceil(d / 256)));
        s.push(small('rms', [B.x, blob(n('ffn_norm')), B.xn], 1));
        s.push(mv(n('router_w'), B.xn, B.rl));
        s.push(small('route', [B.rl, blob(n('router_b')), B.route], 1));
        s.push(mvx(gu, gur, map, B.xn, 0, B.gu)); if (gux) s.push(mvx(gux, null, map, B.xn, 0, B.gu));
        s.push(small('swiglu', [B.gu, blob(n('b_gate_up')), B.route, B.h], Math.ceil(ff / 256), 4));
        s.push(mvx(dn, dnr, map, B.h, ff, B.dn)); if (dnx) s.push(mvx(dnx, null, map, B.h, ff, B.dn));
        s.push(small('combine', [B.x, B.dn, blob(n('b_down')), B.route], Math.ceil(d / 256)));
        this.steps.push(...s);
      }
      this.final = [small('rms', [B.x, blob(g('final_norm')), B.xn], 1), mv(g('lm_head'), B.xn, B.logits)];
      this.eps = cfg.rms_norm_eps || 1e-5; this.limit = cfg.swiglu_limit || 7; this.alpha = cfg.swiglu_alpha || 1.702; this.window = cfg.sliding_window || 128;
      this.posN = 0; this.row = new Float32Array(d);
    }

    reset() { this.posN = 0; }
    setPos(p) { this.posN = p; }
    // one token; want = true returns the logits (Float32Array, valid until the next call)
    async step(tok, want) {
      if (this.posN >= this.ctx) throw new Error(`context window full (${this.ctx} tokens)`);
      const d = this.d, e = this.embed, row = this.row, tab = this.f16, o = tok * d;
      for (let i = 0; i < d; i++) row[i] = tab[e[o + i]];
      const dev = this.dev, q = dev.queue, pos = this.posN, T = pos + 1;
      q.writeBuffer(this.B.x, 0, row);
      const st = new ArrayBuffer(32 + 256), u = new Uint32Array(st), f = new Float32Array(st);
      u[0] = pos; u[1] = 0; u[2] = Math.max(0, T - this.window); u[3] = T; f[4] = this.eps; f[5] = this.limit; f[6] = this.alpha; u[7] = d;
      for (let j = 0; j < 32; j++) { const a = pos * this.inv[j]; f[8 + j] = Math.cos(a) * this.conc; f[40 + j] = Math.sin(a) * this.conc; }
      q.writeBuffer(this.st, 0, st);
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      const run = (s) => { pass.setPipeline(s.p); pass.setBindGroup(0, s.bg); const nx = Math.min(s.n, 65535), nz = Math.ceil(s.n / 65535); pass.dispatchWorkgroups(nx, s.y, nz); };
      for (const s of this.steps) run(s);
      if (want) for (const s of this.final) run(s);
      pass.end();
      if (want) enc.copyBufferToBuffer(this.B.logits, 0, this.B.read, 0, 4 * this.V);
      q.submit([enc.finish()]); this.posN = pos + 1;
      if (!want) return null;
      await this.B.read.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(this.B.read.getMappedRange().slice(0)); this.B.read.unmap();
      return out;
    }
  }
  Lib.GptOssWebGPU = GptOssWebGPU;
  if (typeof module !== 'undefined') module.exports = { GptOssWebGPU };
})();
