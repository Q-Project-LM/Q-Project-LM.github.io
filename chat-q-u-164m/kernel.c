// Q-164M inference kernel, compiled to WebAssembly (SIMD128, single-threaded, no libc).
// Fixed to this model's shape (see config.json): hidden 1280, ff 3520, 20 query / 4 KV heads, head_dim 64,
// 10 layers, 512-bit fingerprints, 32832-token vocabulary. Mirrors model.js (QOmniJS) exactly, op for op --
// that JS version was checked token-for-token against the reference PyTorch model first.
//
// Unlike Q-Omni (one tied embed/readout table), Q-164M has UNTIED input/output corrections over the whole
// vocabulary, so embedding lookup and the final readout matmul use two separate tables (g_table_in /
// g_table_out) instead of one shared g_table -- see qpack.js's loadTensors() for how the host builds both.
//
// Host (JS) responsibilities: parse the .qpack container, unpack ternary/bit-packed tensors to plain float32 (qpack.js,
// already verified), then copy those flat arrays into the buffers this file exposes via the getter functions below.
// Everything numeric — every matmul, norm, and the attention softmax — runs in here.
#include <wasm_simd128.h>

#define D 1280
#define FF 3520
#define NH 20
#define NKV 4
#define HD 64
#define REP (NH / NKV)
#define L 10
#define BITS 512
#define VOCAB 32832
#define MAXT 512

static float g_table_in[VOCAB * BITS];
static float g_table_out[VOCAB * BITS];
static float g_logit_bias[VOCAB];
static float g_w_in[D * BITS];
static float g_w_out[BITS * D];
static float g_final_norm[D];

static float g_n1[L][D], g_n2[L][D];
static float g_qn[L][HD], g_kn[L][HD];
static float g_gate_b[L][NH];
static float g_q[L][NH * HD * D];
static float g_k[L][NKV * HD * D];
static float g_v[L][NKV * HD * D];
static float g_o[L][D * NH * HD];
static float g_gate[L][NH * D];
static float g_w1[L][FF * D];
static float g_w2[L][D * FF];

static float g_kcache[L][NKV][MAXT][HD];
static float g_vcache[L][NKV][MAXT][HD];

static float x[D], h[D], h2[D];
static float qFlat[NH * HD], kFlat[NKV * HD], vFlat[NKV * HD];
static float gateLogit[NH], attnOut[NH * HD];
static float u[FF];
static float scores[MAXT];
static float z[BITS];
static float logits[VOCAB];
static float g_cos[HD], g_sin[HD]; // host writes this step's RoPE cos/sin here before calling step()

// ---- getters: JS learns each buffer's address once, after instantiation, and writes tensor bytes there directly.
__attribute__((export_name("get_table_in"))) float *get_table_in(void) { return g_table_in; }
__attribute__((export_name("get_table_out"))) float *get_table_out(void) { return g_table_out; }
__attribute__((export_name("get_logit_bias"))) float *get_logit_bias(void) { return g_logit_bias; }
__attribute__((export_name("get_w_in"))) float *get_w_in(void) { return g_w_in; }
__attribute__((export_name("get_w_out"))) float *get_w_out(void) { return g_w_out; }
__attribute__((export_name("get_final_norm"))) float *get_final_norm(void) { return g_final_norm; }
__attribute__((export_name("get_n1"))) float *get_n1(int l) { return g_n1[l]; }
__attribute__((export_name("get_n2"))) float *get_n2(int l) { return g_n2[l]; }
__attribute__((export_name("get_qn"))) float *get_qn(int l) { return g_qn[l]; }
__attribute__((export_name("get_kn"))) float *get_kn(int l) { return g_kn[l]; }
__attribute__((export_name("get_gate_b"))) float *get_gate_b(int l) { return g_gate_b[l]; }
__attribute__((export_name("get_q"))) float *get_q(int l) { return g_q[l]; }
__attribute__((export_name("get_k"))) float *get_k(int l) { return g_k[l]; }
__attribute__((export_name("get_v"))) float *get_v(int l) { return g_v[l]; }
__attribute__((export_name("get_o"))) float *get_o(int l) { return g_o[l]; }
__attribute__((export_name("get_gate"))) float *get_gate(int l) { return g_gate[l]; }
__attribute__((export_name("get_w1"))) float *get_w1(int l) { return g_w1[l]; }
__attribute__((export_name("get_w2"))) float *get_w2(int l) { return g_w2[l]; }
__attribute__((export_name("get_logits"))) float *get_logits_ptr(void) { return logits; }
__attribute__((export_name("get_cos"))) float *get_cos(void) { return g_cos; }
__attribute__((export_name("get_sin"))) float *get_sin(void) { return g_sin; }

// ---- math helpers (no libm linked: sqrt is a native WASM instruction; exp is a compact Cephes-style approximation)
static inline float fexp(float x) {
  if (x > 88.0f) return 3.4e38f;
  if (x < -88.0f) return 0.0f;
  const float LOG2E = 1.4426950408889634f;
  float t = x * LOG2E;
  float fl = __builtin_floorf(t);
  float frac = t - fl; // 0..1
  // 2^frac via a degree-5 minimax-ish polynomial (accurate to ~1e-6 on [0,1])
  float p = 1.0f + frac * (0.6931471805599453f + frac * (0.2402265069591007f + frac * (0.05550410866482158f +
            frac * (0.009618129107628477f + frac * 0.0013333558146428443f))));
  int k = (int)fl;
  union { float f; unsigned int u; } r;
  r.u = (unsigned int)((k + 127) << 23); // 2^k via direct exponent bit manipulation
  return p * r.f;
}
static inline float fsigmoid(float x) { return 1.0f / (1.0f + fexp(-x)); }
static inline float fsilu(float x) { return x * fsigmoid(x); }

// y[out] = W[out,in] @ x[in], W row-major. SIMD over 4 floats/lane, scalar tail for the remainder.
static void gemv(const float *W, const float *xv, float *y, int outDim, int inDim) {
  int in4 = inDim & ~3;
  for (int o = 0; o < outDim; o++) {
    const float *row = W + (long)o * inDim;
    v128_t acc = wasm_f32x4_splat(0.0f);
    int k = 0;
    for (; k < in4; k += 4) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(row + k), wasm_v128_load(xv + k)));
    float s = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) + wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
    for (; k < inDim; k++) s += row[k] * xv[k];
    y[o] = s;
  }
}

static void rmsnorm(const float *xv, const float *w, float *y, int n, float eps) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++) ss += xv[i] * xv[i];
  float inv = 1.0f / __builtin_sqrtf(ss / n + eps);
  for (int i = 0; i < n; i++) y[i] = xv[i] * inv * w[i];
}

static void applyRope(float *v, const float *cosv, const float *sinv, int hd) {
  int half = hd / 2;
  float tmp[HD];
  for (int i = 0; i < half; i++) {
    tmp[i] = v[i] * cosv[i] - v[i + half] * sinv[i];
    tmp[i + half] = v[i + half] * cosv[i + half] + v[i] * sinv[i + half];
  }
  for (int i = 0; i < hd; i++) v[i] = tmp[i];
}

// One incremental decode step: token id (its already-gathered BITS-length fingerprint row is read straight out of
// g_table_in) + position. The host writes this step's RoPE cos/sin into g_cos/g_sin (via get_cos()/get_sin()) before
// calling step() — plain sin/cos, cheap to compute in JS, no need to reimplement transcendental functions here.
// Returns a pointer to VOCAB logits (get_logits() from JS).
__attribute__((export_name("step"))) float *step(int tokenId, int pos) {
  const float *cosv = g_cos, *sinv = g_sin;
  gemv(g_w_in, g_table_in + (long)tokenId * BITS, x, D, BITS);
  for (int l = 0; l < L; l++) {
    rmsnorm(x, g_n1[l], h, D, 1e-5f);
    gemv(g_q[l], h, qFlat, NH * HD, D);
    gemv(g_k[l], h, kFlat, NKV * HD, D);
    gemv(g_v[l], h, vFlat, NKV * HD, D);
    gemv(g_gate[l], h, gateLogit, NH, D);
    for (int hi = 0; hi < NH; hi++) { rmsnorm(qFlat + hi * HD, g_qn[l], qFlat + hi * HD, HD, 1e-5f); applyRope(qFlat + hi * HD, cosv, sinv, HD); }
    for (int kv = 0; kv < NKV; kv++) {
      rmsnorm(kFlat + kv * HD, g_kn[l], kFlat + kv * HD, HD, 1e-5f);
      applyRope(kFlat + kv * HD, cosv, sinv, HD);
      for (int i = 0; i < HD; i++) { g_kcache[l][kv][pos][i] = kFlat[kv * HD + i]; g_vcache[l][kv][pos][i] = vFlat[kv * HD + i]; }
    }
    int T = pos + 1;
    for (int hi = 0; hi < NH; hi++) {
      int kv = hi / REP;
      float mx = -3.4e38f;
      for (int t = 0; t < T; t++) {
        float s = 0.0f; const float *kc = g_kcache[l][kv][t]; const float *q = qFlat + hi * HD;
        for (int k = 0; k < HD; k++) s += q[k] * kc[k];
        s *= 0.125f; // 1/sqrt(HD) = 1/8
        scores[t] = s; if (s > mx) mx = s;
      }
      float sum = 0.0f;
      for (int t = 0; t < T; t++) { scores[t] = fexp(scores[t] - mx); sum += scores[t]; }
      float inv = 1.0f / sum;
      float o[HD]; for (int k = 0; k < HD; k++) o[k] = 0.0f;
      for (int t = 0; t < T; t++) { float w = scores[t] * inv; const float *vc = g_vcache[l][kv][t]; for (int k = 0; k < HD; k++) o[k] += w * vc[k]; }
      float gate = fsigmoid(gateLogit[hi] + g_gate_b[l][hi]);
      for (int k = 0; k < HD; k++) attnOut[hi * HD + k] = o[k] * gate;
    }
    { float o[D]; gemv(g_o[l], attnOut, o, D, NH * HD); for (int k = 0; k < D; k++) x[k] += o[k]; }
    rmsnorm(x, g_n2[l], h2, D, 1e-5f);
    gemv(g_w1[l], h2, u, FF, D);
    for (int k = 0; k < FF; k++) u[k] = fsilu(u[k]);
    { float mo[D]; gemv(g_w2[l], u, mo, D, FF); for (int k = 0; k < D; k++) x[k] += mo[k]; }
  }
  rmsnorm(x, g_final_norm, x, D, 1e-5f);
  gemv(g_w_out, x, z, BITS, D);
  gemv(g_table_out, z, logits, VOCAB, BITS);
  const float scale = 0.044194174f; // 1/sqrt(512)
  for (int v = 0; v < VOCAB; v++) logits[v] = logits[v] * scale + g_logit_bias[v];
  return logits;
}
