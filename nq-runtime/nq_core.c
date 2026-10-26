// N-Q decode step in WebAssembly (SIMD128, no libc): the whole forward pass of one token, a port of NQ4JS.step
// (nq_model.js, itself checked against PyTorch). JS only loads tensors (nq_set), feeds tokens (nq_step) and reads logits.
//   clang --target=wasm32 -O3 -msimd128 -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export=__heap_base \
//         -Wl,--export-memory -o nq_kernel.wasm nq_core.c
// Weights: f32 or QPACK2 q8 (int8 rows + float32 per-row scale). All mutable state lives in one block (nq_state_ptr /
// nq_state_bytes) so a snapshot is a single memcpy.
#include <wasm_simd128.h>

typedef unsigned long long u64;
extern unsigned char __heap_base;
void *memset(void *d, int v, unsigned long n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)v; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { unsigned char *p = d; const unsigned char *q = s; while (n--) *p++ = *q++; return d; }

static unsigned long g_top = 0;
__attribute__((export_name("alloc"))) void *alloc(unsigned long bytes) {
  if (!g_top) g_top = ((unsigned long)&__heap_base + 63) & ~63ul;
  unsigned long p = g_top, end = (p + bytes + 63) & ~63ul, have = __builtin_wasm_memory_size(0) * 65536ul;
  if (end > have && __builtin_wasm_memory_grow(0, (end - have + 65535) / 65536) == (unsigned long)-1) return 0;
  g_top = end; memset((void *)p, 0, end - p); return (void *)p;
}

// ------------------------------------------------------------------ math (float32, like the PyTorch reference)
static float fexp(float x) {   // exp with the range reduction 2^k * e^r, |r| <= ln2/2; rel. error ~1e-7
  if (x > 88.f) x = 88.f; if (x < -87.f) return 0.f;
  float k = __builtin_floorf(x * 1.4426950408889634f + 0.5f), r = x - k * 0.6931471805599453f;
  float p = 1.f + r * (1.f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * (0.008333334f + r * (0.0013888889f + r * 0.00019841270f))))));
  union { float f; int i; } u; u.i = ((int)k + 127) << 23; return p * u.f;
}
static float flog1p(float x) {   // log(1+x) for softplus; x >= 0 here
  float y = 1.f + x; union { float f; int i; } u = { y }; int e = ((u.i >> 23) & 255) - 127; u.i = (u.i & 0x7fffff) | 0x3f800000;
  float m = u.f, t = (m - 1.f) / (m + 1.f), t2 = t * t;   // log(m) = 2 atanh(t)
  float lm = 2.f * t * (1.f + t2 * (0.33333334f + t2 * (0.2f + t2 * (0.14285715f + t2 * (0.11111111f + t2 * 0.09090909f)))));
  return lm + (float)e * 0.6931471805599453f;
}
static float fsqrt(float x) { return __builtin_sqrtf(x); }
static float sigm(float v) { return 1.f / (1.f + fexp(-v)); }
static float silu(float v) { return v / (1.f + fexp(-v)); }
static float softplus(float v) { return v > 20.f ? v : flog1p(fexp(v)); }
// sin/cos for RoPE: angle reduction to [-pi, pi] then minimax polynomials
static float fsin_red(float x) { float x2 = x * x; return x * (1.f + x2 * (-0.16666667f + x2 * (0.008333331f + x2 * (-0.00019840874f + x2 * (2.7525562e-6f + x2 * -2.3889859e-8f))))); }
static float fcos_red(float x) { float x2 = x * x; return 1.f + x2 * (-0.5f + x2 * (0.041666638f + x2 * (-0.0013888378f + x2 * (2.4760495e-5f + x2 * -2.6051615e-7f)))); }
static void sincos_(double a, float *s, float *c) {
  double k = __builtin_floor(a / 6.283185307179586 + 0.5); float r = (float)(a - k * 6.283185307179586);
  // reduce further to [-pi/2, pi/2] for accuracy
  float sgn = 1.f; if (r > 1.5707963f) { r = 3.1415927f - r; sgn = -1.f; *s = fsin_red(r); *c = -fcos_red(r); return; }
  if (r < -1.5707963f) { r = -3.1415927f - r; *s = fsin_red(r); *c = -fcos_red(r); return; }
  (void)sgn; *s = fsin_red(r); *c = fcos_red(r);
}

// ------------------------------------------------------------------ tensors
typedef struct { const float *f; const signed char *q; const float *s; int rows, cols; } T;   // f32 or q8
#define MAXL 32
#define NSLOT 40
static T g_t[8 + MAXL * NSLOT];
enum { S_EMBED, S_LMHEAD, S_FIN_NORM, S_FIN_DOWN, S_FIN_UP, S_DEC_W, S_DEC_B };
enum { L_MIX_NORM, L_MIX_DOWN, L_MIX_UP, L_MIX_INJ, L_FFN_NORM, L_FFN_DOWN, L_FFN_UP, L_FFN_INJ, L_GU, L_DOWN,
       L_QKVZ, L_BA, L_CONV, L_ALOG, L_DTB, L_GNORM, L_OUT, L_Q, L_K, L_V, L_O, L_QN, L_KN,
       L_PTAB, L_PKEY, L_PVAL, L_PNK, L_PNQ, L_PNC, L_PCONV, L_PSIZES, L_POFFS, L_PMULT };
static T *ts(int slot) { return &g_t[slot]; }
static T *tl(int l, int id) { return &g_t[8 + l * NSLOT + id]; }
__attribute__((export_name("nq_set"))) void nq_set(int slot, const void *data, const float *scale, int rows, int cols) {
  T *t = &g_t[slot]; t->rows = rows; t->cols = cols;
  if (scale) { t->q = data; t->s = scale; t->f = 0; } else { t->f = data; t->q = 0; t->s = 0; }
}

// q8 x float via integer dot products: x is quantised once per call to int16 (one scale for the vector), then
// i32x4.dot_i16x8_s does 8 multiply-adds per instruction. Partial sums are flushed to float every 256 columns so the
// int32 accumulators cannot overflow (256 * 127 * 32767 < 2^31). Relative error of the x quantisation ~1.5e-5.
static short g_xq[8192];
static void mv_q8_i16(const T *t, const float *x, float *y, int rows, int cols) {
  float mx = 0; for (int c = 0; c < cols; c++) { float a = x[c] < 0 ? -x[c] : x[c]; if (a > mx) mx = a; }
  if (mx == 0) { for (int r = 0; r < rows; r++) y[r] = 0; return; }
  float sx = mx / 32767.f, isx = 32767.f / mx;
  for (int c = 0; c < cols; c++) { float v = x[c] * isx; g_xq[c] = (short)(v < 0 ? v - 0.5f : v + 0.5f); }
  for (int r = 0; r < rows; r++) {
    const signed char *q = t->q + (long)r * cols; float acc = 0; int c = 0;
    while (c + 16 <= cols) {
      int end = c + 256 < cols ? c + 256 : cols; v128_t a0 = wasm_i32x4_splat(0), a1 = a0;
      for (; c + 16 <= end; c += 16) {
        v128_t b = wasm_v128_load(q + c);
        a0 = wasm_i32x4_add(a0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(b), wasm_v128_load(g_xq + c)));
        a1 = wasm_i32x4_add(a1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(b), wasm_v128_load(g_xq + c + 8)));
      }
      v128_t a = wasm_i32x4_add(a0, a1);
      acc += (float)(wasm_i32x4_extract_lane(a, 0) + wasm_i32x4_extract_lane(a, 1)) + (float)(wasm_i32x4_extract_lane(a, 2) + wasm_i32x4_extract_lane(a, 3));
    }
    for (; c < cols; c++) acc += (float)q[c] * (float)g_xq[c];
    y[r] = acc * sx * t->s[r];
  }
}
static int g_int_dot = 1;   // 0 = float path (for checking)
__attribute__((export_name("nq_set_int_dot"))) void nq_set_int_dot(int on) { g_int_dot = on; }

static void mv(const T *t, const float *x, float *y, int rows, int cols) {
  if (t->q && g_int_dot && cols <= 8192) { mv_q8_i16(t, x, y, rows, cols); return; }
  if (t->q) {
    for (int r = 0; r < rows; r++) {
      const signed char *q = t->q + (long)r * cols; v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0; int c = 0;
      for (; c + 16 <= cols; c += 16) {
        v128_t b = wasm_v128_load(q + c), lo = wasm_i16x8_extend_low_i8x16(b), hi = wasm_i16x8_extend_high_i8x16(b);
        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo)), wasm_v128_load(x + c)));
        a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo)), wasm_v128_load(x + c + 4)));
        a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi)), wasm_v128_load(x + c + 8)));
        a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi)), wasm_v128_load(x + c + 12)));
      }
      v128_t a = wasm_f32x4_add(wasm_f32x4_add(a0, a1), wasm_f32x4_add(a2, a3));
      float s = wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1) + wasm_f32x4_extract_lane(a, 2) + wasm_f32x4_extract_lane(a, 3);
      for (; c < cols; c++) s += (float)q[c] * x[c];
      y[r] = s * t->s[r];
    }
  } else {
    for (int r = 0; r < rows; r++) {
      const float *w = t->f + (long)r * cols; v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0; int c = 0;
      for (; c + 16 <= cols; c += 16) {
        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(w + c), wasm_v128_load(x + c)));
        a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_v128_load(w + c + 4), wasm_v128_load(x + c + 4)));
        a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_v128_load(w + c + 8), wasm_v128_load(x + c + 8)));
        a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_v128_load(w + c + 12), wasm_v128_load(x + c + 12)));
      }
      v128_t a = wasm_f32x4_add(wasm_f32x4_add(a0, a1), wasm_f32x4_add(a2, a3));
      float s = wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1) + wasm_f32x4_extract_lane(a, 2) + wasm_f32x4_extract_lane(a, 3);
      for (; c < cols; c++) s += w[c] * x[c];
      y[r] = s;
    }
  }
}
static void row(const T *t, int i, float *out) {
  int n = t->cols;
  if (t->q) { const signed char *q = t->q + (long)i * n; float s = t->s[i]; for (int c = 0; c < n; c++) out[c] = (float)q[c] * s; }
  else { const float *f = t->f + (long)i * n; for (int c = 0; c < n; c++) out[c] = f[c]; }
}
// zero-centred RMSNorm per group of g elements; w has n entries (whole vector) or g entries (shared per group)
static void rmsn(const float *x, const float *w, int n, int g, int wfull, float eps, float *out) {
  for (int g0 = 0; g0 < n; g0 += g) {
    float ss = 0; for (int i = g0; i < g0 + g; i++) ss += x[i] * x[i];
    float inv = 1.f / fsqrt(ss / g + eps);
    for (int i = g0; i < g0 + g; i++) out[i] = x[i] * inv * (1.f + w[wfull ? i : i - g0]);
  }
}

// ------------------------------------------------------------------ config + state
typedef struct { int d, hc, L, rank, ff, V, nk, nv, dk, dv, K, nh, nkv, hd, rot, ple_layer, ple_heads, ple_hpn, ple_hdim, ple_dil, ple_K, eos, max_ctx; float eps, theta; int attn[MAXL]; } Cfg;
static Cfg C;
// state block (snapshot = memcpy of [g_state, g_state + g_state_bytes))
typedef struct { int pos, ple_p1, ple_p2, ple_n; } Head;
static unsigned char *g_state; static unsigned long g_state_bytes;
static Head *H; static float *ple_win;            // (span+1) x HD ring as shift buffer
static float *gdn_win[MAXL], *gdn_S[MAXL];        // (K) x Cconv, nv x dk x dv
static float *kv_k[MAXL], *kv_v[MAXL];            // max_ctx x nkv x hd
// scratch
static float *h, *hn, *low, *gate, *mixed, *inj, *y, *gu, *act, *qkvz, *ba, *xc, *qh, *kh, *o, *qa, *ka, *va, *ao, *sc, *e, *pk, *pv, *pq, *gv, *pc, *fin, *logits, *tmp;
static float g_inv[64];

__attribute__((export_name("nq_cfg"))) int *nq_cfg(void) { return (int *)&C; }   // JS fills the struct, then calls nq_init
__attribute__((export_name("nq_init"))) int nq_init(void) {
  int D = C.d, HD = C.hc * D, Cc = 2 * C.nk * C.dk + C.nv * C.dv;
  unsigned long bytes = sizeof(Head) + 64;
  int span = (C.ple_K - 1) * C.ple_dil;
  bytes += (unsigned long)(span + 1) * HD * 4;
  for (int l = 0; l < C.L; l++) {
    if (C.attn[l]) bytes += 2ul * C.max_ctx * C.nkv * C.hd * 4;
    else bytes += (unsigned long)C.K * Cc * 4 + (unsigned long)C.nv * C.dk * C.dv * 4;
  }
  g_state = alloc(bytes); if (!g_state) return 0; g_state_bytes = bytes;
  unsigned char *p = g_state; H = (Head *)p; p += 64;
  ple_win = (float *)p; p += (unsigned long)(span + 1) * HD * 4;
  for (int l = 0; l < C.L; l++) {
    if (C.attn[l]) { kv_k[l] = (float *)p; p += (unsigned long)C.max_ctx * C.nkv * C.hd * 4; kv_v[l] = (float *)p; p += (unsigned long)C.max_ctx * C.nkv * C.hd * 4; }
    else { gdn_win[l] = (float *)p; p += (unsigned long)C.K * Cc * 4; gdn_S[l] = (float *)p; p += (unsigned long)C.nv * C.dk * C.dv * 4; }
  }
#define A(n) ((float *)alloc(4ul * (n)))
  h = A(HD); hn = A(HD); low = A(C.rank); gate = A(HD); mixed = A(D); inj = A(C.hc); y = A(D); gu = A(2 * C.ff); act = A(C.ff);
  qkvz = A(Cc + C.nv * C.dv); ba = A(2 * C.nv); xc = A(Cc); qh = A(C.nk * C.dk); kh = A(C.nk * C.dk); o = A(C.nv * C.dv);
  qa = A(2 * C.nh * C.hd); ka = A(C.nkv * C.hd); va = A(C.nkv * C.hd); ao = A(C.nh * C.hd); sc = A(C.max_ctx);
  e = A(C.ple_heads * C.ple_hdim + 16); pk = A(HD); pv = A(D); pq = A(HD); gv = A(HD); pc = A(HD); fin = A(D); logits = A(C.V); tmp = A(HD);
  H->ple_p1 = H->ple_p2 = C.eos;
  return (int)bytes;
}
__attribute__((export_name("nq_inv"))) float *nq_inv(void) { return g_inv; }   // RoPE inverse frequencies, filled by the host
__attribute__((export_name("nq_reset"))) void nq_reset(void) { memset(g_state, 0, g_state_bytes); H->ple_p1 = H->ple_p2 = C.eos; }
__attribute__((export_name("nq_state_ptr"))) void *nq_state_ptr(void) { return g_state; }
__attribute__((export_name("nq_state_bytes"))) unsigned long nq_state_bytes(void) { return g_state_bytes; }
__attribute__((export_name("nq_logits"))) float *nq_logits(void) { return logits; }
__attribute__((export_name("nq_hidden"))) float *nq_hidden(void) { return fin; }
__attribute__((export_name("nq_pos"))) int nq_pos(void) { return H->pos; }

// ------------------------------------------------------------------ blocks
static void gres(T *nw, T *down, T *up, T *injw) {   // Gated Residual read: h -> mixed (d) [, inj (hc)]
  int D = C.d, HC = C.hc, HD = HC * D;
  rmsn(h, nw->f, HD, D, 1, C.eps, hn);
  mv(down, hn, low, C.rank, HD);
  for (int i = 0; i < C.rank; i++) low[i] = silu(low[i] / HC);
  mv(up, low, gate, HD, C.rank);
  for (int j = 0; j < D; j++) { float s = 0; for (int c = 0; c < HC; c++) s += sigm(gate[c * D + j]) * hn[c * D + j]; mixed[j] = s / HC; }
  if (injw) { mv(injw, hn, inj, HC, HD); for (int c = 0; c < HC; c++) inj[c] = 2.f * sigm(inj[c] / HC); }
}
static void write_(const float *yv) { int D = C.d; for (int c = 0; c < C.hc; c++) for (int j = 0; j < D; j++) h[c * D + j] += yv[j] * inj[c]; }

static void gdn(int l, const float *x) {
  int D = C.d, nk = C.nk, nv = C.nv, dk = C.dk, dv = C.dv, K = C.K, kd = nk * dk, Cc = 2 * kd + nv * dv, rep = nv / nk;
  mv(tl(l, L_QKVZ), x, qkvz, Cc + nv * dv, D);
  mv(tl(l, L_BA), x, ba, 2 * nv, D);
  float *win = gdn_win[l];
  for (int j = 0; j < K - 1; j++) memcpy(win + (long)j * Cc, win + (long)(j + 1) * Cc, 4ul * Cc);   // shift
  memcpy(win + (long)(K - 1) * Cc, qkvz, 4ul * Cc);
  const float *cw = tl(l, L_CONV)->f;
  for (int ch = 0; ch < Cc; ch++) { float s = 0; for (int j = 0; j < K; j++) s += cw[ch * K + j] * win[(long)j * Cc + ch]; xc[ch] = silu(s); }
  float scale = 1.f / fsqrt((float)dk);
  for (int hI = 0; hI < nk; hI++) {
    float sq = 0, sk = 0; for (int i = 0; i < dk; i++) { sq += xc[hI * dk + i] * xc[hI * dk + i]; sk += xc[kd + hI * dk + i] * xc[kd + hI * dk + i]; }
    float nq = fsqrt(sq), nkk = fsqrt(sk); if (nq < 1e-6f) nq = 1e-6f; if (nkk < 1e-6f) nkk = 1e-6f;
    for (int i = 0; i < dk; i++) { qh[hI * dk + i] = xc[hI * dk + i] / nq * scale; kh[hI * dk + i] = xc[kd + hI * dk + i] / nkk; }
  }
  const float *Al = tl(l, L_ALOG)->f, *dtb = tl(l, L_DTB)->f, *nw = tl(l, L_GNORM)->f;
  float *Sb = gdn_S[l];
  for (int vh = 0; vh < nv; vh++) {
    const float *q = qh + (vh / rep) * dk, *k = kh + (vh / rep) * dk, *v = xc + 2 * kd + vh * dv; float *S = Sb + (long)vh * dk * dv;
    float beta = sigm(ba[vh]), g = fexp(-fexp(Al[vh]) * softplus(ba[nv + vh] + dtb[vh]));
    v128_t gg = wasm_f32x4_splat(g);
    for (int i = 0; i < dk * dv; i += 4) wasm_v128_store(S + i, wasm_f32x4_mul(wasm_v128_load(S + i), gg));
    float kS[256], ov[256];
    for (int j = 0; j < dv; j++) kS[j] = 0;
    for (int i = 0; i < dk; i++) { v128_t ki = wasm_f32x4_splat(k[i]); float *Sr = S + i * dv; for (int j = 0; j < dv; j += 4) wasm_v128_store(kS + j, wasm_f32x4_add(wasm_v128_load(kS + j), wasm_f32x4_mul(ki, wasm_v128_load(Sr + j)))); }
    for (int j = 0; j < dv; j++) kS[j] = (v[j] - kS[j]) * beta;
    for (int j = 0; j < dv; j++) ov[j] = 0;
    for (int i = 0; i < dk; i++) {
      v128_t ki = wasm_f32x4_splat(k[i]), qi = wasm_f32x4_splat(q[i]); float *Sr = S + i * dv;
      for (int j = 0; j < dv; j += 4) {
        v128_t sv = wasm_f32x4_add(wasm_v128_load(Sr + j), wasm_f32x4_mul(ki, wasm_v128_load(kS + j)));
        wasm_v128_store(Sr + j, sv); wasm_v128_store(ov + j, wasm_f32x4_add(wasm_v128_load(ov + j), wasm_f32x4_mul(qi, sv)));
      }
    }
    float ss = 0; for (int j = 0; j < dv; j++) ss += ov[j] * ov[j];
    float inv = 1.f / fsqrt(ss / dv + C.eps);
    for (int j = 0; j < dv; j++) o[vh * dv + j] = ov[j] * inv * (1.f + nw[j]) * sigm(qkvz[Cc + vh * dv + j]);
  }
  mv(tl(l, L_OUT), o, y, D, nv * dv);
}

static void rope_norm(float *v, const float *src, const float *w, const float *cs, const float *sn) {
  int hd = C.hd, half = C.rot / 2; float ss = 0;
  for (int i = 0; i < hd; i++) ss += src[i] * src[i];
  float inv = 1.f / fsqrt(ss / hd + C.eps);
  for (int i = 0; i < hd; i++) v[i] = src[i] * inv * (1.f + w[i]);
  for (int i = 0; i < half; i++) { float a = v[i], b = v[i + half]; v[i] = a * cs[i] - b * sn[i]; v[i + half] = b * cs[i] + a * sn[i]; }
}
static void attn(int l, const float *x) {
  int D = C.d, nh = C.nh, nkv = C.nkv, hd = C.hd, half = C.rot / 2, rep = nh / nkv, pos = H->pos;
  mv(tl(l, L_Q), x, qa, 2 * nh * hd, D); mv(tl(l, L_K), x, ka, nkv * hd, D); mv(tl(l, L_V), x, va, nkv * hd, D);
  float cs[32], sn[32];
  for (int i = 0; i < half; i++) sincos_((double)((float)pos * g_inv[i]), &sn[i], &cs[i]);
  const float *qn = tl(l, L_QN)->f, *kn = tl(l, L_KN)->f;
  if (pos >= C.max_ctx) pos = C.max_ctx - 1;   // (callers keep the context below max_ctx)
  for (int g = 0; g < nkv; g++) { rope_norm(kv_k[l] + ((long)pos * nkv + g) * hd, ka + g * hd, kn, cs, sn); memcpy(kv_v[l] + ((long)pos * nkv + g) * hd, va + g * hd, 4ul * hd); }
  int T = pos + 1; float scl = 1.f / fsqrt((float)hd), qv[256];
  for (int hI = 0; hI < nh; hI++) {
    rope_norm(qv, qa + hI * 2 * hd, qn, cs, sn); int g = hI / rep; float mx = -1e30f;
    for (int t = 0; t < T; t++) { const float *kk = kv_k[l] + ((long)t * nkv + g) * hd; float s = 0; for (int i = 0; i < hd; i++) s += qv[i] * kk[i]; s *= scl; sc[t] = s; if (s > mx) mx = s; }
    float sum = 0; for (int t = 0; t < T; t++) { sc[t] = fexp(sc[t] - mx); sum += sc[t]; }
    float *out = ao + hI * hd; for (int i = 0; i < hd; i++) out[i] = 0;
    for (int t = 0; t < T; t++) { const float *vv = kv_v[l] + ((long)t * nkv + g) * hd; float w = sc[t] / sum; for (int i = 0; i < hd; i++) out[i] += w * vv[i]; }
    for (int i = 0; i < hd; i++) out[i] *= sigm(qa[hI * 2 * hd + hd + i]);
  }
  mv(tl(l, L_O), ao, y, D, nh * hd);
}

static void ffn(int l, const float *x) {
  int ff = C.ff; mv(tl(l, L_GU), x, gu, 2 * ff, C.d);
  for (int i = 0; i < ff; i++) act[i] = silu(gu[i]) * gu[ff + i];
  mv(tl(l, L_DOWN), act, y, C.d, ff);
}

static void ple(int l, int tok) {
  int D = C.d, HC = C.hc, HD = HC * D, nh = C.ple_heads, hpn = C.ple_hpn, hdim = C.ple_hdim, eos = C.eos;
  const long long *sizes = (const long long *)tl(l, L_PSIZES)->f, *offs = (const long long *)tl(l, L_POFFS)->f, *mult = (const long long *)tl(l, L_PMULT)->f;
  u64 sh[3] = { (u64)tok, (u64)H->ple_p1, (u64)(H->ple_p1 == eos ? eos : H->ple_p2) };
  int j = 0;
  for (int ng = 2; ng <= 3; ng++) {
    u64 hh = sh[0] * (u64)mult[0];
    for (int s = 1; s < ng; s++) hh ^= sh[s] * (u64)mult[s];
    for (int k = (ng - 2) * hpn; k < (ng - 1) * hpn; k++, j++) { long id = (long)(hh % (u64)sizes[k]) + (long)offs[k]; row(tl(l, L_PTAB), (int)id, e + j * hdim); }
  }
  H->ple_p2 = H->ple_p1; H->ple_p1 = tok;
  int ne = nh * hdim;
  mv(tl(l, L_PKEY), e, tmp, HD, ne); rmsn(tmp, tl(l, L_PNK)->f, HD, D, 1, C.eps, pk);
  mv(tl(l, L_PVAL), e, pv, D, ne);
  rmsn(h, tl(l, L_PNQ)->f, HD, D, 1, C.eps, pq);
  for (int c = 0; c < HC; c++) {
    float s = 0; for (int i = 0; i < D; i++) s += pk[c * D + i] * pq[c * D + i];
    s /= fsqrt((float)D); float a = s < 0 ? -s : s; if (a < 1e-6f) a = 1e-6f; s = fsqrt(a) * (s > 0 ? 1.f : s < 0 ? -1.f : 0.f);
    float gg = sigm(s); for (int i = 0; i < D; i++) gv[c * D + i] = gg * pv[i];
  }
  int span = (C.ple_K - 1) * C.ple_dil;
  for (int k = 0; k < span; k++) memcpy(ple_win + (long)k * HD, ple_win + (long)(k + 1) * HD, 4ul * HD);   // shift, newest last
  rmsn(gv, tl(l, L_PNC)->f, HD, D, 1, C.eps, ple_win + (long)span * HD);
  int n = H->ple_n < span + 1 ? ++H->ple_n : span + 1;   // valid entries (zeros before the start == conv zero padding)
  (void)n;
  const float *cw = tl(l, L_PCONV)->f; int K = C.ple_K;
  for (int ch = 0; ch < HD; ch++) {
    float s = 0; for (int jj = 0; jj < K; jj++) s += cw[ch * K + jj] * ple_win[(long)(span - (K - 1 - jj) * C.ple_dil) * HD + ch];
    h[ch] += gv[ch] + silu(s);
  }
}

// one token; want: 0 = state only, 1 = logits, 2 = final hidden (JEV head input). Returns the pointer to the result.
__attribute__((export_name("nq_step"))) float *nq_step(int tok, int want) {
  int D = C.d, HC = C.hc; float *er = tmp;
  row(ts(S_EMBED), tok, er);
  for (int c = 0; c < HC; c++) memcpy(h + c * D, er, 4ul * D);
  for (int l = 0; l < C.L; l++) {
    if (l == C.ple_layer) ple(l, tok);
    gres(tl(l, L_MIX_NORM), tl(l, L_MIX_DOWN), tl(l, L_MIX_UP), tl(l, L_MIX_INJ));
    if (C.attn[l]) attn(l, mixed); else gdn(l, mixed);
    write_(y);
    gres(tl(l, L_FFN_NORM), tl(l, L_FFN_DOWN), tl(l, L_FFN_UP), tl(l, L_FFN_INJ));
    ffn(l, mixed); write_(y);
  }
  if (C.ple_layer < 0) { H->ple_p2 = H->ple_p1; H->ple_p1 = tok; }
  H->pos++;
  if (!want) return 0;
  gres(ts(S_FIN_NORM), ts(S_FIN_DOWN), ts(S_FIN_UP), 0);
  memcpy(fin, mixed, 4ul * D);
  if (want == 2) return fin;
  mv(ts(S_LMHEAD), fin, logits, C.V, D);
  return logits;
}

// top-k candidates by logit (for sampling without sorting the whole vocabulary): fills idx[0..k) in descending order
__attribute__((export_name("nq_topk"))) int nq_topk(const float *x, int n, int k, int *idx) {
  int m = 0;
  for (int i = 0; i < n; i++) {
    float v = x[i]; if (m == k && v <= x[idx[m - 1]]) continue;
    int p = m < k ? m++ : k - 1; while (p > 0 && x[idx[p - 1]] < v) { idx[p] = idx[p - 1]; p--; } idx[p] = i;
  }
  return m;
}
