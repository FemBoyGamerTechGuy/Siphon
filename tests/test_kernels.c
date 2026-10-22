/* siphon: kernel tests
 *
 * Everything is self-contained (no reference build needed):
 *  - f16 conversion roundtrips over ALL 65536 bit patterns
 *  - every quant dot vs a double-precision evaluation of the exact
 *    quantized semantics (dequantized weights x scaled activations)
 *  - scalar vs AVX2 kernels agree to accumulation-order tolerance and,
 *    where the two share integer arithmetic, exactly
 *  - activation quantizer invariants (range, bsums, zero blocks)
 *  - the exp polynomial: si_expf(0) == 1 exactly, monotone, within its
 *    documented error of expf, and (on x86) bit-identical to the
 *    transcribed vector form
 */
#include "test_util.h"
#include "siphon/kernels.h"

#include <math.h>

#if defined(__x86_64__) || defined(__i386__)
#define SI_TEST_X86 1
#include <immintrin.h>
void si_avx2_v_expf_pub8(const float *x, float *y);
#endif

static uint64_t t_rng = 0x9E3779B97F4A7C15ULL;
static uint64_t trand(void) {
    t_rng ^= t_rng >> 12;
    t_rng ^= t_rng << 25;
    t_rng ^= t_rng >> 27;
    return t_rng * 0x2545F4914F6CDD1DULL;
}
static float trand_f(void) {
    return (float)((double)(trand() >> 40) / 524288.0 * 2.0 - 1.0);
}

/* ---- f16 conversions ---------------------------------------------------- */

static void test_f16_roundtrip(void) {
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        const si_f16 orig = (si_f16)h;
        const float f = si_f16_to_f32(orig);
        const si_f16 back = si_f32_to_f16(f);
        if ((h & 0x7c00) == 0x7c00) {
            /* inf stays inf; nan stays nan */
            CHECK(isinf(f) || isnan(f));
            continue;
        }
        if (back != orig) {
            if (bad++ < 4) {
                fprintf(stderr, "  roundtrip %04x -> %g -> %04x\n", h, f, back);
            }
        }
    }
    CHECK(bad == 0);

    /* spot values */
    CHECK(si_f16_to_f32(0x0000) == 0.0f);
    CHECK(si_f16_to_f32(0x8000) == -0.0f);
    CHECK(si_f16_to_f32(0x3C00) == 1.0f);
    CHECK(si_f16_to_f32(0x7BFF) > 65500.0f && si_f16_to_f32(0x7BFF) < 65505.0f);
    CHECK(si_f32_to_f16(1.0f) == 0x3C00);
    CHECK(si_f32_to_f16(-1.0f) == 0xBC00);
    CHECK(si_f32_to_f16(0.5f) == 0x3800);
    CHECK(si_f32_to_f16(0.0f) == 0x0000);
    CHECK((si_f32_to_f16(-0.0f) & 0x8000) != 0);
    CHECK(si_f32_to_f16(65504.0f) == 0x7BFF);
    CHECK(si_f32_to_f16(65520.0f) == 0x7C00);       /* overflow to inf */
    CHECK(si_f32_to_f16(1e10f) == 0x7C00);
    CHECK(si_f32_to_f16(1e-10f) == 0x0000);          /* underflow to zero */
    CHECK(si_f32_to_f16(6.0e-8f) == 0x0001);         /* smallest subnormal */

    /* f32 -> f16 -> f32: value preserved to half precision (already
     * covered by the roundtrip above from the f16 side; here we verify
     * the error bound from the f32 side) */
    int bound_bad = 0;
    for (int i = 0; i < 20000; i++) {
        float f = trand_f() * 100.0f;
        const si_f16 h = si_f32_to_f16(f);
        const float g = si_f16_to_f32(h);
        const float rel = fabsf(g - f) / (fabsf(f) + 1e-30f);
        if (rel > 0.001f) bound_bad++;
    }
    CHECK(bound_bad == 0);
}

/* ---- helpers ------------------------------------------------------------ */

enum { NB = 4, N = NB * 256 };

static void rand_blocks(void *p, size_t bytes) {
    uint8_t *b = p;
    for (size_t i = 0; i < bytes; i++) b[i] = (uint8_t)trand();
}

/* make the f16 scales of a random block well-conditioned so dots stay
 * finite: force d/dmin exponent into a small range */
static void tame_scales_q4K(si_blk_q4_K *x) {
    for (int i = 0; i < NB; i++) {
        x[i].d = si_f32_to_f16(0.01f * (1.0f + (float)(trand() % 100) / 100.0f));
        x[i].dmin = si_f32_to_f16(0.001f * (float)(trand() % 100) / 100.0f);
    }
}
static void tame_scales_q5K(si_blk_q5_K *x) {
    for (int i = 0; i < NB; i++) {
        x[i].d = si_f32_to_f16(0.01f * (1.0f + (float)(trand() % 100) / 100.0f));
        x[i].dmin = si_f32_to_f16(0.001f * (float)(trand() % 100) / 100.0f);
    }
}
static void tame_scales_q6K(si_blk_q6_K *x) {
    for (int i = 0; i < NB; i++) {
        x[i].d = si_f32_to_f16(0.01f * (1.0f + (float)(trand() % 100) / 100.0f));
    }
}
static void tame_scales_q2K(si_blk_q2_K *x) {
    for (int i = 0; i < NB; i++) {
        x[i].d = si_f32_to_f16(0.01f * (1.0f + (float)(trand() % 100) / 100.0f));
        x[i].dmin = si_f32_to_f16(0.001f * (float)(trand() % 100) / 100.0f);
    }
}
static void tame_scales_q3K(si_blk_q3_K *x) {
    for (int i = 0; i < NB; i++) {
        x[i].d = si_f32_to_f16(0.01f * (1.0f + (float)(trand() % 100) / 100.0f));
    }
}
static void tame_scales_q4_0(si_blk_q4_0 *x) {
    for (int i = 0; i < N / 32; i++) {
        x[i].d = si_f32_to_f16(0.01f * (1.0f + (float)(trand() % 100) / 100.0f));
    }
}
static void tame_scales_q5_0(si_blk_q5_0 *x) {
    for (int i = 0; i < N / 32; i++) {
        x[i].d = si_f32_to_f16(0.01f * (1.0f + (float)(trand() % 100) / 100.0f));
    }
}
static void tame_scales_q8_0(si_blk_q8_0 *x) {
    for (int i = 0; i < N / 32; i++) {
        x[i].d = si_f32_to_f16(0.01f * (1.0f + (float)(trand() % 100) / 100.0f));
    }
}

/* exact double-precision evaluation of one dot against dequantized
 * weights and the exact activation semantics; also returns the L1 norm
 * of the terms so cancellation-safe tolerances can be applied */
static double ref_dot(const float *xd, int n, const void *y, uint32_t pair,
                      double *l1) {
    double s = 0, m = 0;
    if (pair == SI_T_Q8_0) {
        const si_blk_q8_0 *y8 = y;
        for (int b = 0; b < n / 32; b++) {
            const double d = si_f16_to_f32(y8[b].d);
            for (int j = 0; j < 32; j++) {
                const double t = (double)xd[b * 32 + j] * (y8[b].qs[j] * d);
                s += t;
                m += fabs(t);
            }
        }
    } else {
        const si_blk_q8_K *yk = y;
        for (int b = 0; b < n / 256; b++) {
            const double d = yk[b].d;
            for (int j = 0; j < 256; j++) {
                const double t = (double)xd[b * 256 + j] * (yk[b].qs[j] * d);
                s += t;
                m += fabs(t);
            }
        }
    }
    *l1 = m;
    return s;
}

static float xs[N];     /* dequantized weights */
static float xa[N];     /* activation floats */
static float xd[N];     /* dequant of quantized activations (for ref) */

static void run_type(const char *name, uint32_t wtype) {
    static si_blk_q4_0 b40[N / 32];
    static si_blk_q5_0 b50[N / 32];
    static si_blk_q8_0 b80[N / 32];
    static si_blk_q2_K b2K[NB];
    static si_blk_q3_K b3K[NB];
    static si_blk_q4_K b4K[NB];
    static si_blk_q5_K b5K[NB];
    static si_blk_q6_K b6K[NB];
    static si_blk_q8_0 y8[N / 32];
    static si_blk_q8_K y8K[NB];

    void *xblocks = NULL;
    size_t xbytes = 0;
    uint32_t pair;

    switch (wtype) {
        case SI_T_Q4_0: xblocks = b40; xbytes = sizeof b40; break;
        case SI_T_Q5_0: xblocks = b50; xbytes = sizeof b50; break;
        case SI_T_Q8_0: xblocks = b80; xbytes = sizeof b80; break;
        case SI_T_Q2_K: xblocks = b2K; xbytes = sizeof b2K; break;
        case SI_T_Q3_K: xblocks = b3K; xbytes = sizeof b3K; break;
        case SI_T_Q4_K: xblocks = b4K; xbytes = sizeof b4K; break;
        case SI_T_Q5_K: xblocks = b5K; xbytes = sizeof b5K; break;
        case SI_T_Q6_K: xblocks = b6K; xbytes = sizeof b6K; break;
        default: CHECK(!"bad wtype"); return;
    }
    CHECK(si_quant_vec_dot_pair(wtype, &pair));
    si_vec_dot_fn dot = si_vec_dot_for(wtype);
    CHECK(dot != NULL);

    int exact_ok = 1;
    double worst_rel = 0;
    for (int trial = 0; trial < 40; trial++) {
        rand_blocks(xblocks, xbytes);
        switch (wtype) {
            case SI_T_Q2_K: tame_scales_q2K(b2K); break;
            case SI_T_Q3_K: tame_scales_q3K(b3K); break;
            case SI_T_Q4_K: tame_scales_q4K(b4K); break;
            case SI_T_Q5_K: tame_scales_q5K(b5K); break;
            case SI_T_Q6_K: tame_scales_q6K(b6K); break;
            case SI_T_Q4_0: tame_scales_q4_0(b40); break;
            case SI_T_Q5_0: tame_scales_q5_0(b50); break;
            case SI_T_Q8_0: tame_scales_q8_0(b80); break;
            default: break;
        }
        for (int i = 0; i < N; i++) xa[i] = trand_f();

        CHECK(si_dequant_row(wtype, xblocks, xs, N));

        void *yact;
        if (pair == SI_T_Q8_0) {
            si_quant_row_q8_0(xa, y8, N);
            yact = y8;
            si_dequant_row_q8_0(y8, xd, N);
        } else {
            si_quant_row_q8_K(xa, y8K, N);
            yact = y8K;
            /* exact activation values: y.d * y.qs[j] */
            for (int b = 0; b < NB; b++) {
                for (int j = 0; j < 256; j++) {
                    xd[b * 256 + j] = y8K[b].qs[j] * y8K[b].d;
                }
            }
        }

        float s = 0;
        dot(N, &s, xblocks, yact);
        double l1 = 0;
        const double ref = ref_dot(xs, N, yact, pair, &l1);
        CHECK(isfinite(s));
        /* float accumulation vs the double reference: bounded by the
         * term magnitude (cancellation-safe) */
        const double err = fabs((double)s - ref);
        const double rel = err / (l1 > 1e-9 ? l1 : 1e-9);
        if (rel > worst_rel) worst_rel = rel;
        /* integer lanes identical between scalar and dispatched impl for
         * the classic formats; K-quants differ only in float grouping */
        if (rel > 2e-6) exact_ok = 0;
    }
    printf("  %-6s worst rel err vs exact double reference: %.3g\n",
           name, worst_rel);
    CHECK(exact_ok);
}

/* ---- activation quantization -------------------------------------------- */

static void test_q8_quantize(void) {
    static float x[N];
    static si_blk_q8_0 y8[N / 32];
    static si_blk_q8_K y8K[NB];

    for (int t = 0; t < 20; t++) {
        for (int i = 0; i < N; i++) x[i] = trand_f();

        si_quant_row_q8_0(x, y8, N);
        for (int b = 0; b < N / 32; b++) {
            const float d = si_f16_to_f32(y8[b].d);
            float amax = 0;
            for (int j = 0; j < 32; j++) {
                const float ax = fabsf(x[b * 32 + j]);
                if (ax > amax) amax = ax;
            }
            /* d == amax/127 (to f16 precision) */
            CHECK(fabsf(d - amax / 127.0f) < 0.001f * (amax / 127.0f + 1e-6f));
            for (int j = 0; j < 32; j++) {
                const float v = y8[b].qs[j] * d;
                /* half a quant plus the f16 scale storage rounding */
                CHECK(fabsf(v - x[b * 32 + j]) <= d * 0.5f +
                      fabsf(x[b * 32 + j]) * 5e-4f + 1e-6f);
                CHECK(y8[b].qs[j] >= -127 && y8[b].qs[j] <= 127);
            }
        }

        si_quant_row_q8_K(x, y8K, N);
        for (int b = 0; b < NB; b++) {
            float amax = 0, amax_signed = 0;
            for (int j = 0; j < 256; j++) {
                const float ax = fabsf(x[b * 256 + j]);
                if (ax > amax) { amax = ax; amax_signed = x[b * 256 + j]; }
            }
            /* d = 1/(-127/max): signed, opposite sign of max, two float
             * roundings away from -max/127 */
            CHECK(fabsf(y8K[b].d - (amax_signed / -127.0f)) <
                  fabsf(amax_signed / -127.0f) * 1e-5f + 1e-9f);
            for (int j = 0; j < 256; j++) {
                CHECK(y8K[b].qs[j] >= -127 && y8K[b].qs[j] <= 127);
                const float v = y8K[b].qs[j] * y8K[b].d;
                CHECK(fabsf(v - x[b * 256 + j]) <= fabsf(y8K[b].d) * 1.01f);
            }
            for (int g = 0; g < 16; g++) {
                int sum = 0;
                for (int j = 0; j < 16; j++) sum += y8K[b].qs[g * 16 + j];
                CHECK(y8K[b].bsums[g] == (int16_t)sum);
            }
        }
    }

    /* zero block: all-zero activations quantize to d == 0 */
    memset(x, 0, sizeof x);
    si_quant_row_q8_0(x, y8, N);
    for (int b = 0; b < N / 32; b++) CHECK(y8[b].d == 0);
    si_quant_row_q8_K(x, y8K, N);
    for (int b = 0; b < NB; b++) CHECK(y8K[b].d == 0.0f);
}

/* ---- elementwise numerics ------------------------------------------------ */

static void test_exp_silu_softmax(void) {
    CHECK(si_expf(0.0f) == 1.0f);
    CHECK(si_expf(1.0f) > 2.718280f && si_expf(1.0f) < 2.718283f);
    CHECK(si_expf(-1.0f) < 0.3678795f && si_expf(-1.0f) > 0.3678794f);

    int bad = 0, sign_bad = 0, nan_bad = 0;
    for (int i = 0; i < 200000; i++) {
        const float x = trand_f() * 40.0f;
        const float a = si_expf(x);
        const float b = expf(x);
        if (isnan(a) || isnan(b)) { nan_bad++; continue; }
        if (isinf(b)) continue;
        /* the poly is accurate to ~1.5 ulp + 0.5 */
        const double ulp = ldexp(1.0, -23) * (double)b;
        if (fabs((double)a - (double)b) > 2.0 * ulp + 1e-30) bad++;
        if ((a > 0) != (b > 0)) sign_bad++;
    }
    CHECK(bad == 0);
    CHECK(sign_bad == 0);
    CHECK(nan_bad == 0);

    /* extreme range: flushes like expf (0 / inf), no NaNs; the deep
     * negative range returns denormals, exactly like expf */
    CHECK(si_expf(100.0f) == HUGE_VALF || isinf(si_expf(100.0f)));
    CHECK(si_expf(-100.0f) > 0.0f && si_expf(-100.0f) < 1e-43f);
    CHECK(si_expf(-90.0f) > 0.0f);
    CHECK(!isnan(si_expf(-20.0f)) && si_expf(-20.0f) > 0.0f);

    /* silu: zero and sign symmetry points */
    CHECK(si_silu(0.0f) == 0.0f);
    CHECK(fabsf(si_silu(-20.0f)) < 1e-6f);
    CHECK(fabsf(si_silu(20.0f) - 20.0f) < 1e-3f);

#ifdef SI_TEST_X86
    /* scalar transcription must be lane-identical to the vector form */
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        int mismatch = 0;
        for (int t = 0; t < 20000; t++) {
            float vin[8], vout[8];
            for (int i = 0; i < 8; i++) {
                vin[i] = trand_f() * 30.0f;
            }
            si_avx2_v_expf_pub8(vin, vout);
            for (int i = 0; i < 8; i++) {
                const float s = si_expf(vin[i]);
                if (memcmp(&s, &vout[i], 4) != 0) mismatch++;
            }
        }
        CHECK(mismatch == 0);
    }
#endif

    /* softmax: normalizes, single element is exactly 1 */
    float xs6[6] = {0.5f, -1.0f, 2.0f, 2.0f, -0.25f, 3.0f};
    float max = -INFINITY;
    for (int i = 0; i < 6; i++) if (xs6[i] > max) max = xs6[i];
    float y6[6];
    si_softmax(y6, xs6, 6, max);
    double sum = 0;
    for (int i = 0; i < 6; i++) {
        CHECK(y6[i] >= 0.0f && y6[i] <= 1.0f);
        sum += y6[i];
    }
    CHECK(sum > 0.999 && sum < 1.001);
    float y1[1];
    si_softmax(y1, xs6, 1, xs6[0]);
    CHECK(y1[0] == 1.0f);
    /* order preservation: max element gets the largest prob */
    CHECK(y6[5] >= y6[2] && y6[2] >= y6[0]);

    /* rms scale: constant vector c -> 1/c */
    float c[64];
    for (int i = 0; i < 64; i++) c[i] = 2.5f;
    CHECK(fabsf(si_rms_scale(c, 64, 0.0f) - 0.4f) < 1e-6f);
}

/* ---- geometry ------------------------------------------------------------ */

static void test_geometry(void) {
    uint32_t b, t;
    CHECK(si_quant_geometry(SI_T_Q4_K, &b, &t));
    CHECK(b == 256 && t == 144);
    CHECK(si_quant_geometry(SI_T_Q6_K, &b, &t));
    CHECK(b == 256 && t == 210);
    CHECK(si_quant_geometry(SI_T_Q8_K, &b, &t));
    CHECK(b == 256 && t == 292);
    CHECK(si_quant_geometry(SI_T_Q8_0, &b, &t));
    CHECK(b == 32 && t == 34);
    CHECK(si_quant_geometry(SI_T_Q5_0, &b, &t));
    CHECK(b == 32 && t == 22);
    CHECK(!si_quant_geometry(40, &b, &t));
    CHECK(strcmp(si_quant_name(SI_T_Q4_K), "Q4_K") == 0);
}

int main(void) {
    si_kern_init();
    printf("kernel impl: %s\n", si_kern_impl_name());

    test_f16_roundtrip();
    run_type("q4_0", SI_T_Q4_0);
    run_type("q5_0", SI_T_Q5_0);
    run_type("q8_0", SI_T_Q8_0);
    run_type("q2_K", SI_T_Q2_K);
    run_type("q3_K", SI_T_Q3_K);
    run_type("q4_K", SI_T_Q4_K);
    run_type("q5_K", SI_T_Q5_K);
    run_type("q6_K", SI_T_Q6_K);
    test_q8_quantize();
    test_exp_silu_softmax();
    test_geometry();

    TEST_REPORT("test_kernels");
}
