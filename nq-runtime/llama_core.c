// Llama-family decode step in WebAssembly (SIMD128, no libc): Llama / Qwen2 / Mistral / SmolLM style decoders
// (RMSNorm, RoPE rotate-half, GQA, optional q/k/v bias, SwiGLU, tied or untied head). Weights f32, QPACK2 q8 or q4
// (groups of 32 columns: 16 bytes, byte j = column j low nibble / column j+16 high nibble, value - 8, one scale per group).
//   clang --target=wasm32 -O3 -msimd128 -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export=__heap_base \
//         -Wl,--export-memory -Wl,--max-memory=4294967296 -o llama_core.wasm llama_core.c
#include <wasm_simd128.h>

extern unsigned char __heap_base;
void *memset(void *d, int v, unsigned long n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)v; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { unsigned char *p = d; const unsigned char *q = s; while (n--) *p++ = *q++; return d; }
static unsigned long g_top = 0;
__attribute__((export_name("alloc"))) void *alloc(unsigned long bytes) {
  if (!g_top) g_top = ((unsigned long)&__heap_base + 63) & ~63ul;
  unsigned long p = g_top, end = (p + bytes + 63) & ~63ul, have = __builtin_wasm_memory_size(0) * 65536ul;
  if (end > have && __builtin_wasm_memory_grow(0, (end - have + 65535) / 65536) == (unsigned long)-1) return 0;
  g_top = end; return (void *)p;
}

static float fexp(float x) {
  if (x > 88.f) x = 88.f; if (x < -87.f) return 0.f;
  float k = __builtin_floorf(x * 1.4426950408889634f + 0.5f), r = x - k * 0.6931471805599453f;
  float p = 1.f + r * (1.f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * (0.008333334f + r * (0.0013888889f + r * 0.00019841270f))))));
  union { float f; int i; } u; u.i = ((int)k + 127) << 23; return p * u.f;
}
static float silu(float v) { return v / (1.f + fexp(-v)); }
static float fsin_red(float x) { float x2 = x * x; return x * (1.f + x2 * (-0.16666667f + x2 * (0.008333331f + x2 * (-0.00019840874f + x2 * (2.7525562e-6f + x2 * -2.3889859e-8f))))); }
static float fcos_red(float x) { float x2 = x * x; return 1.f + x2 * (-0.5f + x2 * (0.041666638f + x2 * (-0.0013888378f + x2 * (2.4760495e-5f + x2 * -2.6051615e-7f)))); }
static void sincos_(double a, float *s, float *c) {
  double k = __builtin_floor(a / 6.283185307179586 + 0.5); float r = (float)(a - k * 6.283185307179586);
  if (r > 1.5707963f) { r = 3.1415927f - r; *s = fsin_red(r); *c = -fcos_red(r); return; }
  if (r < -1.5707963f) { r = -3.1415927f - r; *s = fsin_red(r); *c = -fcos_red(r); return; }
  *s = fsin_red(r); *c = fcos_red(r);
}

typedef struct { const float *f; const signed char *q; const float *s; const unsigned char *q4; int rows, cols; } T;
#define MAXL 128
#define NSLOT 16
enum { S_EMBED, S_LMHEAD, S_FNORM };
enum { L_ANORM, L_FNORM, L_WQ, L_WK, L_WV, L_WO, L_BQ, L_BK, L_BV, L_GATE, L_UP, L_DOWN };
static T g_t[8 + MAXL * NSLOT];
static T *ts(int s) { return &g_t[s]; }
static T *tl(int l, int id) { return &g_t[8 + l * NSLOT + id]; }
__attribute__((export_name("ll_set"))) void ll_set(int slot, const void *data, const float *scale, int rows, int cols) {
  T *t = &g_t[slot]; t->rows = rows; t->cols = cols; t->q4 = 0;
  if (scale) { t->q = data; t->s = scale; t->f = 0; } else { t->f = data; t->q = 0; t->s = 0; }
}
__attribute__((export_name("ll_set4"))) void ll_set4(int slot, const void *data, const float *scale, int rows, int cols) {
  T *t = &g_t[slot]; t->rows = rows; t->cols = cols; t->q4 = data; t->s = scale; t->f = 0; t->q = 0;
}

static short *g_xq; static int g_xq_n;
// q4 x float (same as nq_core.c): int16 activations, each 32-column group unpacked to int16 and dotted, times its scale
static void mv_q4(const T *t, const float *x, float *y, int rows, int cols) {
  float mx = 0; for (int c = 0; c < cols; c++) { float a = x[c] < 0 ? -x[c] : x[c]; if (a > mx) mx = a; }
  if (mx == 0) { for (int r = 0; r < rows; r++) y[r] = 0; return; }
  float sx = mx / 32767.f, isx = 32767.f / mx;
  for (int c = 0; c < cols; c++) { float v = x[c] * isx; g_xq[c] = (short)(v < 0 ? v - 0.5f : v + 0.5f); }
  const int G = cols / 32; const v128_t m = wasm_i8x16_splat(15), e = wasm_i8x16_splat(8);
  for (int r = 0; r < rows; r++) {
    const unsigned char *q = t->q4 + (long)r * (cols / 2); const float *s = t->s + (long)r * G; v128_t acc = wasm_f32x4_splat(0);
    for (int k = 0; k < G; k++) {
      v128_t b = wasm_v128_load(q + 16 * k), lo = wasm_i8x16_sub(wasm_v128_and(b, m), e), hi = wasm_i8x16_sub(wasm_u8x16_shr(b, 4), e);
      const short *xk = g_xq + 32 * k;
      v128_t d = wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(lo), wasm_v128_load(xk)), wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(lo), wasm_v128_load(xk + 8)));
      d = wasm_i32x4_add(d, wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(hi), wasm_v128_load(xk + 16)), wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(hi), wasm_v128_load(xk + 24))));
      acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(d), wasm_f32x4_splat(s[k])));
    }
    y[r] = (wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) + wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3)) * sx;
  }
}
static void mv(const T *t, const float *x, float *y, int rows, int cols) {
  if (t->q4) { mv_q4(t, x, y, rows, cols); return; }
  if (t->q) {   // int16 activations x int8 weights (see nq_core.c)
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
    return;
  }
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
static void row(const T *t, int i, float *out) {
  int n = t->cols;
  if (t->q4) {
    const unsigned char *q = t->q4 + (long)i * (n / 2); const float *s = t->s + (long)i * (n / 32);
    for (int k = 0; k < n / 32; k++) for (int j = 0; j < 16; j++) { unsigned char b = q[16 * k + j]; out[32 * k + j] = (float)((int)(b & 15) - 8) * s[k]; out[32 * k + j + 16] = (float)((int)(b >> 4) - 8) * s[k]; }
  }
  else if (t->q) { const signed char *q = t->q + (long)i * n; float s = t->s[i]; for (int c = 0; c < n; c++) out[c] = (float)q[c] * s; }
  else { const float *f = t->f + (long)i * n; for (int c = 0; c < n; c++) out[c] = f[c]; }
}
static void rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
  float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.f / __builtin_sqrtf(ss / n + eps);
  for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

typedef struct { int d, L, nh, nkv, hd, ff, V, max_ctx, tie, bias; float eps, theta; } Cfg;
static Cfg C;
static int g_pos;
static float *kc, *vc;                 // [L][max_ctx][nkv*hd]
static float *x, *xn, *q, *k, *v, *att, *o, *g, *u, *sc, *logits, g_inv[256];
__attribute__((export_name("ll_cfg"))) int *ll_cfg(void) { return (int *)&C; }
__attribute__((export_name("ll_inv"))) float *ll_inv(void) { return g_inv; }
__attribute__((export_name("ll_init"))) int ll_init(void) {
  unsigned long kv = (unsigned long)C.L * C.max_ctx * C.nkv * C.hd * 4;
  kc = alloc(kv); vc = alloc(kv); if (!kc || !vc) return 0;
  int big = C.d > C.ff ? C.d : C.ff; if (C.nh * C.hd > big) big = C.nh * C.hd;
#define A(n) ((float *)alloc(4ul * (n)))
  x = A(C.d); xn = A(C.d); q = A(C.nh * C.hd); k = A(C.nkv * C.hd); v = A(C.nkv * C.hd); att = A(C.nh * C.hd); o = A(C.d);
  g = A(C.ff); u = A(C.ff); sc = A(C.max_ctx); logits = A(C.V); g_xq = (short *)alloc(2ul * (big + 16)); g_xq_n = big;
  if (!logits || !g_xq) return 0;
  g_pos = 0; return 1;
}
__attribute__((export_name("ll_reset"))) void ll_reset(void) { g_pos = 0; }
__attribute__((export_name("ll_pos"))) int ll_pos(void) { return g_pos; }
__attribute__((export_name("ll_set_pos"))) void ll_set_pos(int p) { g_pos = p; }
__attribute__((export_name("ll_kv"))) float *ll_kv(int which) { return which ? vc : kc; }   // for snapshots (host copies the used prefix)

static void rope(float *vec, const float *cs, const float *sn) {
  int half = C.hd / 2;
  for (int i = 0; i < half; i++) { float a = vec[i], b = vec[i + half]; vec[i] = a * cs[i] - b * sn[i]; vec[i + half] = b * cs[i] + a * sn[i]; }
}

// one token; want: 0 = state only, 1 = logits, 2 = final hidden. Returns a pointer to the result.
__attribute__((export_name("ll_step"))) float *ll_step(int tok, int want) {
  int D = C.d, nh = C.nh, nkv = C.nkv, hd = C.hd, rep = nh / nkv, pos = g_pos, kvd = nkv * hd, half = hd / 2;
  if (pos >= C.max_ctx) return 0;
  row(ts(S_EMBED), tok, x);
  float cs[128], sn[128];
  for (int i = 0; i < half; i++) sincos_((double)pos * (double)g_inv[i], &sn[i], &cs[i]);
  float scl = 1.f / __builtin_sqrtf((float)hd);
  for (int l = 0; l < C.L; l++) {
    rmsnorm(x, tl(l, L_ANORM)->f, D, C.eps, xn);
    mv(tl(l, L_WQ), xn, q, nh * hd, D); mv(tl(l, L_WK), xn, k, kvd, D); mv(tl(l, L_WV), xn, v, kvd, D);
    if (C.bias) { const float *bq = tl(l, L_BQ)->f, *bk = tl(l, L_BK)->f, *bv = tl(l, L_BV)->f;
      for (int i = 0; i < nh * hd; i++) q[i] += bq[i]; for (int i = 0; i < kvd; i++) { k[i] += bk[i]; v[i] += bv[i]; } }
    for (int hI = 0; hI < nh; hI++) rope(q + hI * hd, cs, sn);
    for (int gI = 0; gI < nkv; gI++) rope(k + gI * hd, cs, sn);
    float *kl = kc + ((long)l * C.max_ctx) * kvd, *vl = vc + ((long)l * C.max_ctx) * kvd;
    memcpy(kl + (long)pos * kvd, k, 4ul * kvd); memcpy(vl + (long)pos * kvd, v, 4ul * kvd);
    int Tn = pos + 1;
    for (int hI = 0; hI < nh; hI++) {
      const float *qh = q + hI * hd; int gI = hI / rep; float mx = -1e30f;
      for (int t = 0; t < Tn; t++) {
        const float *kk = kl + (long)t * kvd + gI * hd; v128_t a = wasm_f32x4_splat(0);
        for (int i = 0; i < hd; i += 4) a = wasm_f32x4_add(a, wasm_f32x4_mul(wasm_v128_load(qh + i), wasm_v128_load(kk + i)));
        float s = (wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1) + wasm_f32x4_extract_lane(a, 2) + wasm_f32x4_extract_lane(a, 3)) * scl;
        sc[t] = s; if (s > mx) mx = s;
      }
      float sum = 0; for (int t = 0; t < Tn; t++) { sc[t] = fexp(sc[t] - mx); sum += sc[t]; }
      float *out = att + hI * hd; for (int i = 0; i < hd; i++) out[i] = 0;
      float inv = 1.f / sum;
      for (int t = 0; t < Tn; t++) {
        const float *vv = vl + (long)t * kvd + gI * hd; v128_t w = wasm_f32x4_splat(sc[t] * inv);
        for (int i = 0; i < hd; i += 4) wasm_v128_store(out + i, wasm_f32x4_add(wasm_v128_load(out + i), wasm_f32x4_mul(w, wasm_v128_load(vv + i))));
      }
    }
    mv(tl(l, L_WO), att, o, D, nh * hd);
    for (int i = 0; i < D; i++) x[i] += o[i];
    rmsnorm(x, tl(l, L_FNORM)->f, D, C.eps, xn);
    mv(tl(l, L_GATE), xn, g, C.ff, D); mv(tl(l, L_UP), xn, u, C.ff, D);
    for (int i = 0; i < C.ff; i++) g[i] = silu(g[i]) * u[i];
    mv(tl(l, L_DOWN), g, o, D, C.ff);
    for (int i = 0; i < D; i++) x[i] += o[i];
  }
  g_pos = pos + 1;
  if (!want) return 0;
  rmsnorm(x, ts(S_FNORM)->f, D, C.eps, xn);
  if (want == 2) return xn;
  mv(C.tie ? ts(S_EMBED) : ts(S_LMHEAD), xn, logits, C.V, D);
  return logits;
}
