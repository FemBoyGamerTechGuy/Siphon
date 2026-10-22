/* siphon: numerical kernels
 *
 * Dot products, quantization and dequantization for every weight format
 * Siphon supports. Two implementations exist per kernel: a portable
 * scalar reference and, when the CPU allows, an AVX2+FMA+F16C variant
 * selected once at startup via CPUID (SI_SCALAR=1 in the environment
 * forces the scalar path for debugging and for proving the two agree).
 *
 * The AVX2 variants are transcribed structure-for-structure from ggml's
 * x86 kernels (lane grouping, maddubs/madd pairing, scale shuffles, f32
 * FMA accumulation and horizontal-sum order included) so results are
 * bit-identical to the reference implementation, not merely close.
 *
 * Elementwise ops (exp/silu/softmax) use ggml's polynomial transcribed
 * scalar with fmaf, including its range-correction branches, which is
 * lane-identical to the vector form.
 */
#ifndef SIPHON_KERNELS_H
#define SIPHON_KERNELS_H

#include "siphon/quant.h"

/* ---- f16 conversions (both round-to-nearest-even, exact like F16C) ---- */
float si_f16_to_f32(si_f16 h);
si_f16 si_f32_to_f16(float f);

/* ---- dispatch state ---- */
typedef enum {
    SI_KERN_SCALAR = 0,
    SI_KERN_AVX2   = 1,
} si_kern_impl;

/* Called once from main/test entry points; reads CPUID, honors SI_SCALAR. */
void si_kern_init(void);
si_kern_impl si_kern_impl_get(void);
/* capability string for logs, e.g. "avx2+fma+f16c" or "scalar". */
const char *si_kern_impl_name(void);

/* ---- dot products ---------------------------------------------------
 * s = <x, y> where x is a row of `wtype`-encoded blocks and y is the
 * matching activation format (see si_quant_vec_dot_pair).
 * n must be a multiple of the weight block size.
 * F32 rows use plain float x/y; F16 rows use si_f16 x/y.
 * ------------------------------------------------------------------- */
typedef void (*si_vec_dot_fn)(int n, float *s, const void *x, const void *y);
/* Returns NULL for unsupported weight types. */
si_vec_dot_fn si_vec_dot_for(uint32_t wtype);

/* Individual kernels (used directly by tests; dispatch is per wtype). */
void si_dot_f32(int n, float *s, const float *x, const float *y);
void si_dot_f16(int n, float *s, const si_f16 *x, const si_f16 *y);
void si_dot_q4_0_q8_0(int n, float *s, const si_blk_q4_0 *x, const si_blk_q8_0 *y);
void si_dot_q5_0_q8_0(int n, float *s, const si_blk_q5_0 *x, const si_blk_q8_0 *y);
void si_dot_q8_0_q8_0(int n, float *s, const si_blk_q8_0 *x, const si_blk_q8_0 *y);
void si_dot_q2_K_q8_K(int n, float *s, const si_blk_q2_K *x, const si_blk_q8_K *y);
void si_dot_q3_K_q8_K(int n, float *s, const si_blk_q3_K *x, const si_blk_q8_K *y);
void si_dot_q4_K_q8_K(int n, float *s, const si_blk_q4_K *x, const si_blk_q8_K *y);
void si_dot_q5_K_q8_K(int n, float *s, const si_blk_q5_K *x, const si_blk_q8_K *y);
void si_dot_q6_K_q8_K(int n, float *s, const si_blk_q6_K *x, const si_blk_q8_K *y);

/* ---- activation quantization ---------------------------------------
 * Both bit-exact with the reference: q8_0 uses the SIMD round-to-nearest
 * scheme, q8_K uses nearest_int() with the -127/max scale and bsums.
 * n must be a multiple of the respective block size.
 * ------------------------------------------------------------------- */
void si_quant_row_q8_0(const float *x, si_blk_q8_0 *y, int n);
void si_quant_row_q8_K(const float *x, si_blk_q8_K *y, int n);

/* ---- dequantization (reference semantics, used for embeddings) ----- */
void si_dequant_row_f16(const si_f16 *x, float *y, int n);
void si_dequant_row_q4_0(const si_blk_q4_0 *x, float *y, int n);
void si_dequant_row_q5_0(const si_blk_q5_0 *x, float *y, int n);
void si_dequant_row_q8_0(const si_blk_q8_0 *x, float *y, int n);
void si_dequant_row_q2_K(const si_blk_q2_K *x, float *y, int n);
void si_dequant_row_q3_K(const si_blk_q3_K *x, float *y, int n);
void si_dequant_row_q4_K(const si_blk_q4_K *x, float *y, int n);
void si_dequant_row_q5_K(const si_blk_q5_K *x, float *y, int n);
void si_dequant_row_q6_K(const si_blk_q6_K *x, float *y, int n);
/* Dispatch by ggml type; false for unsupported. */
bool si_dequant_row(uint32_t type, const void *x, float *y, int n);

/* ---- elementwise numerics (ggml-exact) ------------------------------ */
/* ggml's vector exp polynomial, transcribed scalar with fmaf and the
 * same range-correction branches; bit-identical per lane to v_expf. */
float si_expf(float x);
/* silu via the polynomial: x / (1 + si_expf(-x)). */
float si_silu(float x);
/* One softmax row, ggml's order: 8-wide exp groups accumulated into a
 * double, scalar tail via expf, then y *= 1/sum. max must be the row
 * max (first occurrence semantics irrelevant to the result). */
void si_softmax(float *y, const float *x, int n, float max);
/* rmsnorm scale factor: 1/sqrt(mean + eps), mean over f32 squares
 * accumulated in double (ggml semantics). */
float si_rms_scale(const float *x, int n, float eps);

#endif /* SIPHON_KERNELS_H */
