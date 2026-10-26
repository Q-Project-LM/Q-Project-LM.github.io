// N-Q matrix-vector kernels for the browser runtime, compiled to WebAssembly with SIMD128 (no libc).
//   clang --target=wasm32 -O3 -msimd128 -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export=__heap_base -Wl,--export-memory \
//         -o nq_kernel.wasm nq_kernel.c
// The host (nq_model.js) copies each weight matrix into linear memory once and then calls these per token.
// A decode step is a long series of y = W x products (FFN, projections, Gated Residual, lm_head): memory-bound, so
// SIMD + int8 weights (QPACK2 'q8': per-row scale, y_r = scale_r * sum_c q_rc x_c) are where the speed comes from.
#include <wasm_simd128.h>

extern unsigned char __heap_base;
// no libc: the compiler may still emit calls to these
void *memset(void *d, int v, unsigned long n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)v; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { unsigned char *p = d; const unsigned char *q = s; while (n--) *p++ = *q++; return d; }
static unsigned long g_top = 0;

__attribute__((export_name("alloc"))) void *alloc(unsigned long bytes) {
  if (!g_top) g_top = ((unsigned long)&__heap_base + 63) & ~63ul;
  unsigned long p = g_top, end = (p + bytes + 63) & ~63ul;
  unsigned long have = __builtin_wasm_memory_size(0) * 65536ul;
  if (end > have && __builtin_wasm_memory_grow(0, (end - have + 65535) / 65536) == (unsigned long)-1) return 0;
  g_top = end;
  return (void *)p;
}

// y[r] = sum_c W[r*cols + c] * x[c]   (W float32, row-major)
__attribute__((export_name("matvec_f32"))) void matvec_f32(const float *W, const float *x, float *y, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const float *w = W + (long)r * cols;
    v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;
    int c = 0;
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

// int8 weights, float activations: 16 weights per load, widened to four f32x4 lanes
__attribute__((export_name("matvec_q8"))) void matvec_q8(const signed char *Q, const float *scale, const float *x, float *y, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const signed char *q = Q + (long)r * cols;
    v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
      v128_t b = wasm_v128_load(q + c);
      v128_t lo = wasm_i16x8_extend_low_i8x16(b), hi = wasm_i16x8_extend_high_i8x16(b);
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo)), wasm_v128_load(x + c)));
      a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo)), wasm_v128_load(x + c + 4)));
      a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi)), wasm_v128_load(x + c + 8)));
      a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi)), wasm_v128_load(x + c + 12)));
    }
    v128_t a = wasm_f32x4_add(wasm_f32x4_add(a0, a1), wasm_f32x4_add(a2, a3));
    float s = wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1) + wasm_f32x4_extract_lane(a, 2) + wasm_f32x4_extract_lane(a, 3);
    for (; c < cols; c++) s += (float)q[c] * x[c];
    y[r] = s * scale[r];
  }
}

// one row of a q8 table (embedding / n-gram table lookups), dequantised to float
__attribute__((export_name("row_q8"))) void row_q8(const signed char *Q, const float *scale, int row, int cols, float *out) {
  const signed char *q = Q + (long)row * cols; float s = scale[row];
  for (int c = 0; c < cols; c++) out[c] = (float)q[c] * s;
}
