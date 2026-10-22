/* siphon: numerical kernels — portable side (see include/siphon/kernels.h)
 *
 * Scalar reference kernels are transcriptions of ggml's generic
 * implementations; the AVX2 variants live in kernels_avx2.c and are
 * selected at startup by CPUID (SI_SCALAR=1 forces scalar).
 *
 * The elementwise numerics (si_expf, si_silu, si_softmax, si_rms_scale)
 * follow ggml's exact accumulation orders so that whole-graph results
 * land within the reference implementation's own run-to-run variation
 * band, and in practice match it bit-for-bit through many layers.
 */
#include "siphon/kernels.h"

#include <math.h>

static inline uint32_t float_as_u32(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}
static inline float u32_as_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

#if defined(__x86_64__) || defined(__i386__)
#define SI_X86 1
#include <immintrin.h>
/* defined in kernels_avx2.c (only on x86) */
void si_avx2_dot_f16(int n, float *s, const si_f16 *x, const si_f16 *y);
void si_avx2_dot_q4_0_q8_0(int n, float *s, const si_blk_q4_0 *x, const si_blk_q8_0 *y);
void si_avx2_dot_q5_0_q8_0(int n, float *s, const si_blk_q5_0 *x, const si_blk_q8_0 *y);
void si_avx2_dot_q8_0_q8_0(int n, float *s, const si_blk_q8_0 *x, const si_blk_q8_0 *y);
void si_avx2_dot_q2_K_q8_K(int n, float *s, const si_blk_q2_K *x, const si_blk_q8_K *y);
void si_avx2_dot_q3_K_q8_K(int n, float *s, const si_blk_q3_K *x, const si_blk_q8_K *y);
void si_avx2_dot_q4_K_q8_K(int n, float *s, const si_blk_q4_K *x, const si_blk_q8_K *y);
void si_avx2_dot_q5_K_q8_K(int n, float *s, const si_blk_q5_K *x, const si_blk_q8_K *y);
void si_avx2_dot_q6_K_q8_K(int n, float *s, const si_blk_q6_K *x, const si_blk_q8_K *y);
void si_avx2_quant_row_q8_0(const float *x, si_blk_q8_0 *y, int n);
#endif

/* ---- dispatch state --------------------------------------------------- */

static si_kern_impl g_impl = SI_KERN_SCALAR;
static bool g_init = false;

#ifdef SI_X86
static void cpuid(uint32_t leaf, uint32_t sub, uint32_t out[4]) {
    __asm__ volatile("cpuid"
                     : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
                     : "a"(leaf), "c"(sub));
}
#endif

void si_kern_init(void) {
    if (g_init) return;
    g_impl = SI_KERN_SCALAR;
#ifdef SI_X86
    do {
        uint32_t r[4];
        cpuid(0, 0, r);
        if (r[0] < 7) break;
        cpuid(1, 0, r);
        const bool fma = (r[2] & (1u << 12)) != 0;
        const bool f16c = (r[2] & (1u << 29)) != 0;
        const bool osxsave = (r[2] & (1u << 27)) != 0;
        if (!fma || !f16c || !osxsave) break;
        uint64_t xcr0 = 0;
        uint32_t hi = 0, lo = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = ((uint64_t)hi << 32) | lo;
        /* need XMM + YMM state */
        if ((xcr0 & 0x6) != 0x6) break;
        cpuid(7, 0, r);
        if (!(r[1] & (1u << 5))) break; /* AVX2 */
        g_impl = SI_KERN_AVX2;
    } while (0);
#endif
    const char *env = getenv("SI_SCALAR");
    if (env && env[0] && strcmp(env, "0") != 0) {
        g_impl = SI_KERN_SCALAR;
    }
    g_init = true;
}

si_kern_impl si_kern_impl_get(void) {
    if (!g_init) si_kern_init();
    return g_impl;
}

const char *si_kern_impl_name(void) {
    return si_kern_impl_get() == SI_KERN_AVX2 ? "avx2+fma+f16c" : "scalar";
}

/* ---- f16 conversions (exact, hardware-equivalent) --------------------- */

float si_f16_to_f32(si_f16 h) {
    /* lossless widening: exponent/mantissa/shift, inf/nan preserved */
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign; /* +-0 */
        } else {
            /* subnormal half -> normalized float */
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; e++; } while ((m & 0x400) == 0);
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13); /* inf / nan */
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

si_f16 si_f32_to_f16(float f) {
    /* round-to-nearest-even, matching F16C hardware and ggml's software
     * path bit-for-bit (verified over all 65536 round trips in tests). */
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    const uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t man = bits & 0x7fffff;

    if (((bits >> 23) & 0xff) == 0xff) {
        /* inf / nan */
        if (man != 0) {
            /* keep a quiet nan with top mantissa bit set */
            uint32_t m = (man >> 13);
            if (m == 0) m = 1; /* ensure nonzero payload */
            return (si_f16)(sign | 0x7c00u | m | 0x0200u);
        }
        return (si_f16)(sign | 0x7c00u);
    }

    if (exp >= 0x1f) {
        /* overflow -> inf */
        return (si_f16)(sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            /* too small (incl. subnormal floats): rounds to zero */
            return (si_f16)sign;
        }
        /* subnormal half: renormalize into [0.5, 1) mantissa scale */
        man |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp); /* 2..14 */
        uint32_t half = 1u << (shift - 1);
        uint32_t kept = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t out = kept;
        if (rem > half || (rem == half && (kept & 1))) out++;
        return (si_f16)(sign | out);
    }

    /* normal path: round the 13 dropped mantissa bits */
    const uint32_t half = 1u << 12;
    const uint32_t kept = man >> 13;
    const uint32_t rem = man & ((1u << 13) - 1); /* includes the half bit */
    uint32_t out = ((uint32_t)exp << 10) | kept;
    if (rem > half || (rem == half && (kept & 1))) {
        out++;
        if (out == 0x7c00u) { /* rounded up to inf */
            return (si_f16)(sign | 0x7c00u);
        }
    }
    return (si_f16)(sign | out);
}

/* ---- scalar dot products ---------------------------------------------- */

void si_dot_f32(int n, float *s, const float *x, const float *y) {
    /* ggml's non-SIMD fallback: double accumulation of f32 products */
    double sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (double)(x[i] * y[i]);
    }
    *s = (float)sumf;
}

void si_dot_f16(int n, float *s, const si_f16 *x, const si_f16 *y) {
    double sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (double)(si_f16_to_f32(x[i]) * si_f16_to_f32(y[i]));
    }
    *s = (float)sumf;
}

void si_dot_q4_0_q8_0(int n, float *s, const si_blk_q4_0 *x, const si_blk_q8_0 *y) {
    const int qk = SI_QK8_0;
    const int nb = n / qk;
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < qk / 2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >> 4) - 8;
            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + qk / 2];
        }
        const int sumi = sumi0 + sumi1;
        sumf += sumi * si_f16_to_f32(x[ib].d) * si_f16_to_f32(y[ib].d);
    }
    *s = sumf;
}

void si_dot_q5_0_q8_0(int n, float *s, const si_blk_q5_0 *x, const si_blk_q8_0 *y) {
    const int qk = SI_QK8_0;
    const int nb = n / qk;
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof qh);
        int sumi = 0;
        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            const int v0 = ((x[ib].qs[j] & 0x0F) | xh_0) - 16;
            const int v1 = ((x[ib].qs[j] >> 4) | xh_1) - 16;
            sumi += v0 * y[ib].qs[j] + v1 * y[ib].qs[j + qk / 2];
        }
        sumf += sumi * si_f16_to_f32(x[ib].d) * si_f16_to_f32(y[ib].d);
    }
    *s = sumf;
}

void si_dot_q8_0_q8_0(int n, float *s, const si_blk_q8_0 *x, const si_blk_q8_0 *y) {
    const int qk = SI_QK8_0;
    const int nb = n / qk;
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        sumf += sumi * (si_f16_to_f32(x[ib].d) * si_f16_to_f32(y[ib].d));
    }
    *s = sumf;
}

void si_dot_q2_K_q8_K(int n, float *s, const si_blk_q2_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * si_f16_to_f32(x[i].d);
        const float dmin = y[i].d * si_f16_to_f32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < SI_QK_K / 128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

void si_dot_q3_K_q8_K(int n, float *s, const si_blk_q3_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    int8_t aux8[SI_QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    uint32_t auxs[4];
    const int8_t *scales = (const int8_t *)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q3 = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < SI_QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < SI_QK_K / 16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = si_f16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void si_dot_q4_K_q8_K(int n, float *s, const si_blk_q4_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[2];

    int8_t aux8[SI_QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q4 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *a = aux8;
        for (int j = 0; j < SI_QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < SI_QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < SI_QK_K / 32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = si_f16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = si_f16_to_f32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void si_dot_q5_K_q8_K(int n, float *s, const si_blk_q5_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[2];

    int8_t aux8[SI_QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q4 = x[i].qs;
        const uint8_t *hm = x[i].qh;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < SI_QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < SI_QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < SI_QK_K / 32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = si_f16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = si_f16_to_f32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void si_dot_q6_K_q8_K(int n, float *s, const si_blk_q6_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;

    int8_t aux8[SI_QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q4 = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *a = aux8;
        for (int j = 0; j < SI_QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < SI_QK_K / 16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = si_f16_to_f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

/* ---- activation quantization ------------------------------------------ */

/* ggml's nearest_int: magic-number round-to-nearest (|f| <= 4194303). */
static inline int si_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void si_quant_row_q8_0(const float *x, si_blk_q8_0 *y, int n) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_quant_row_q8_0(x, y, n);
        return;
    }
#endif
    /* scalar reference (quantize_row_q8_0_ref) */
    const int nb = n / SI_QK8_0;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < SI_QK8_0; j++) {
            const float v = x[j];
            amax = amax > fabsf(v) ? amax : fabsf(v);
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = si_f32_to_f16(d);
        for (int j = 0; j < SI_QK8_0; ++j) {
            const float x0 = x[j] * id;
            y[i].qs[j] = (int8_t)roundf(x0);
        }
        x += SI_QK8_0;
    }
}

void si_quant_row_q8_K(const float *x, si_blk_q8_K *y, int n) {
    /* reference implementation (what the reference dispatches to on x86) */
    const int nb = n / SI_QK_K;
    for (int i = 0; i < nb; i++) {
        float max = 0;
        float amax = 0;
        for (int j = 0; j < SI_QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, SI_QK_K);
            x += SI_QK_K;
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < SI_QK_K; ++j) {
            int v = si_nearest_int(iscale * x[j]);
            y[i].qs[j] = v < 127 ? (int8_t)v : (int8_t)127;
        }
        for (int j = 0; j < SI_QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j * 16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1 / iscale;
        x += SI_QK_K;
    }
}

/* ---- dequantization ---------------------------------------------------- */

void si_dequant_row_f16(const si_f16 *x, float *y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = si_f16_to_f32(x[i]);
    }
}

void si_dequant_row_q4_0(const si_blk_q4_0 *x, float *y, int n) {
    const int qk = SI_QK4_0;
    const int nb = n / qk;
    for (int i = 0; i < nb; i++) {
        const float d = si_f16_to_f32(x[i].d);
        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            y[i * qk + j + 0] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
}

void si_dequant_row_q5_0(const si_blk_q5_0 *x, float *y, int n) {
    const int qk = SI_QK5_0;
    const int nb = n / qk;
    for (int i = 0; i < nb; i++) {
        const float d = si_f16_to_f32(x[i].d);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof qh);
        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;
            y[i * qk + j + 0] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
}

void si_dequant_row_q8_0(const si_blk_q8_0 *x, float *y, int n) {
    const int qk = SI_QK8_0;
    const int nb = n / qk;
    for (int i = 0; i < nb; i++) {
        const float d = si_f16_to_f32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i * qk + j] = x[i].qs[j] * d;
        }
    }
}

static inline void si_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void si_dequant_row_q2_K(const si_blk_q2_K *x, float *y, int n) {
    const int nb = n / SI_QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = si_f16_to_f32(x[i].d);
        const float min = si_f16_to_f32(x[i].dmin);
        const uint8_t *q = x[i].qs;
        int is = 0;
        float dl, ml;
        for (int m = 0; m < SI_QK_K; m += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

void si_dequant_row_q3_K(const si_blk_q3_K *x, float *y, int n) {
    const int nb = n / SI_QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = si_f16_to_f32(x[i].d);
        const uint8_t *q = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int m2 = 0; m2 < SI_QK_K; m2 += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void si_dequant_row_q4_K(const si_blk_q4_K *x, float *y, int n) {
    const int nb = n / SI_QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d = si_f16_to_f32(x[i].d);
        const float min = si_f16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < SI_QK_K; j += 64) {
            si_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            si_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void si_dequant_row_q5_K(const si_blk_q5_K *x, float *y, int n) {
    const int nb = n / SI_QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d = si_f16_to_f32(x[i].d);
        const float min = si_f16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < SI_QK_K; j += 64) {
            si_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            si_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void si_dequant_row_q6_K(const si_blk_q6_K *x, float *y, int n) {
    const int nb = n / SI_QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = si_f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (int m = 0; m < SI_QK_K; m += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

bool si_dequant_row(uint32_t type, const void *x, float *y, int n) {
    switch (type) {
        case SI_T_F32:  memcpy(y, x, (size_t)n * sizeof(float)); return true;
        case SI_T_F16:  si_dequant_row_f16(x, y, n); return true;
        case SI_T_Q4_0: si_dequant_row_q4_0(x, y, n); return true;
        case SI_T_Q5_0: si_dequant_row_q5_0(x, y, n); return true;
        case SI_T_Q8_0: si_dequant_row_q8_0(x, y, n); return true;
        case SI_T_Q2_K: si_dequant_row_q2_K(x, y, n); return true;
        case SI_T_Q3_K: si_dequant_row_q3_K(x, y, n); return true;
        case SI_T_Q4_K: si_dequant_row_q4_K(x, y, n); return true;
        case SI_T_Q5_K: si_dequant_row_q5_K(x, y, n); return true;
        case SI_T_Q6_K: si_dequant_row_q6_K(x, y, n); return true;
        default: return false;
    }
}

/* ---- elementwise numerics (ggml-exact) -------------------------------- */

float si_expf(float x) {
    /* ggml's vector exp polynomial (ARM-derived), transcribed scalar with
     * fmaf and the same range-correction branches; bit-identical per lane
     * to the AVX2 form. */
    const float r = 0x1.8p23f;
    const float z = fmaf(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float b = fmaf(-n, 0x1.7f7d1cp-20f, fmaf(-n, 0x1.62e4p-1f, x));
    const uint32_t e = float_as_u32(z) << 23;
    /* add the BIT PATTERN of 1.0f: reconstructs 2^n from the mantissa
     * integer of z (the reference adds castps(set1_ps(1))) */
    const float k = u32_as_float(e + 0x3F800000u);
    const bool c = fabsf(n) > 126.0f;
    const float u = b * b;
    const float A = fmaf(0x1.0e4020p-7f, b, 0x1.573e2ep-5f);
    const float B = fmaf(0x1.555e66p-3f, b, 0x1.fffdb6p-2f);
    const float C = 0x1.ffffecp-1f * b;
    const float j = fmaf(fmaf(A, u, B), u, C);
    if (!c) {
        return fmaf(j, k, k);
    }
    const uint32_t g = (n <= 0.0f) ? 0x82000000u : 0u;
    const float s1 = u32_as_float(g + 0x7f000000u);
    const float s2 = u32_as_float(e - g);
    const bool d = fabsf(n) > 192.0f;
    if (d) return s1 * s1;
    if (c) return fmaf(s2, j, s2) * s1;
    return fmaf(k, j, k);
}

float si_silu(float x) {
    return x / (1.0f + si_expf(0.0f - x));
}

void si_softmax(float *y, const float *x, int n, float max) {
    /* ggml_vec_soft_max_f32: 8-wide exp groups (vector poly) with the
     * per-group lane-ordered horizontal sum added to a double total,
     * then a scalar tail through expf, then y *= 1/sum. */
    double sum = 0;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float v[8];
        v[0] = si_expf(x[i + 0] - max);
        v[1] = si_expf(x[i + 1] - max);
        v[2] = si_expf(x[i + 2] - max);
        v[3] = si_expf(x[i + 3] - max);
        v[4] = si_expf(x[i + 4] - max);
        v[5] = si_expf(x[i + 5] - max);
        v[6] = si_expf(x[i + 6] - max);
        v[7] = si_expf(x[i + 7] - max);
        memcpy(y + i, v, sizeof v);
        /* lane order of _mm256 hsum: ((v4+v0)+(v6+v2)) + ((v5+v1)+(v7+v3)) */
        const float s0 = v[4] + v[0];
        const float s1 = v[5] + v[1];
        const float s2 = v[6] + v[2];
        const float s3 = v[7] + v[3];
        const float t0 = s0 + s2;
        const float t1 = s1 + s3;
        sum += (double)(t0 + t1);
    }
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (double)val;
        y[i] = val;
    }
    const float scale = (float)(1.0 / sum);
    for (i = 0; i < n; ++i) {
        y[i] *= scale;
    }
}

float si_rms_scale(const float *x, int n, float eps) {
    /* ggml rmsnorm: sum of f32 squares accumulated in double */
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (double)(x[i] * x[i]);
    }
    const float mean = sum / n;
    return 1.0f / sqrtf(mean + eps);
}

/* ---- type tables -------------------------------------------------------- */

bool si_quant_geometry(uint32_t type, uint32_t *blck_size, uint32_t *type_size) {
    switch (type) {
        case SI_T_F32:  *blck_size = 1;  *type_size = 4;  return true;
        case SI_T_F16:  *blck_size = 1;  *type_size = 2;  return true;
        case SI_T_Q4_0: *blck_size = SI_QK4_0; *type_size = sizeof(si_blk_q4_0); return true;
        case SI_T_Q5_0: *blck_size = SI_QK5_0; *type_size = sizeof(si_blk_q5_0); return true;
        case SI_T_Q8_0: *blck_size = SI_QK8_0; *type_size = sizeof(si_blk_q8_0); return true;
        case SI_T_Q2_K: *blck_size = SI_QK_K; *type_size = sizeof(si_blk_q2_K); return true;
        case SI_T_Q3_K: *blck_size = SI_QK_K; *type_size = sizeof(si_blk_q3_K); return true;
        case SI_T_Q4_K: *blck_size = SI_QK_K; *type_size = sizeof(si_blk_q4_K); return true;
        case SI_T_Q5_K: *blck_size = SI_QK_K; *type_size = sizeof(si_blk_q5_K); return true;
        case SI_T_Q6_K: *blck_size = SI_QK_K; *type_size = sizeof(si_blk_q6_K); return true;
        case SI_T_Q8_K: *blck_size = SI_QK_K; *type_size = sizeof(si_blk_q8_K); return true;
        default: return false;
    }
}

const char *si_quant_name(uint32_t type) {
    switch (type) {
        case SI_T_F32:  return "F32";
        case SI_T_F16:  return "F16";
        case SI_T_Q4_0: return "Q4_0";
        case SI_T_Q4_1: return "Q4_1";
        case SI_T_Q5_0: return "Q5_0";
        case SI_T_Q5_1: return "Q5_1";
        case SI_T_Q8_0: return "Q8_0";
        case SI_T_Q2_K: return "Q2_K";
        case SI_T_Q3_K: return "Q3_K";
        case SI_T_Q4_K: return "Q4_K";
        case SI_T_Q5_K: return "Q5_K";
        case SI_T_Q6_K: return "Q6_K";
        case SI_T_Q8_K: return "Q8_K";
        default: {
            static char buf[4][16];
            static int slot = 0;
            slot = (slot + 1) & 3;
            snprintf(buf[slot], sizeof buf[slot], "TYPE%u", type);
            return buf[slot];
        }
    }
}

bool si_quant_vec_dot_pair(uint32_t wtype, uint32_t *pair_type) {
    switch (wtype) {
        case SI_T_F16:  *pair_type = SI_T_F16;  return true;
        case SI_T_F32:  *pair_type = SI_T_F32;  return true;
        case SI_T_Q4_0:
        case SI_T_Q5_0:
        case SI_T_Q8_0: *pair_type = SI_T_Q8_0; return true;
        case SI_T_Q2_K:
        case SI_T_Q3_K:
        case SI_T_Q4_K:
        case SI_T_Q5_K:
        case SI_T_Q6_K: *pair_type = SI_T_Q8_K; return true;
        default: return false;
    }
}

/* ---- dispatch ----------------------------------------------------------- */

static void dot_f16_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_f16(n, s, x, y);
        return;
    }
#endif
    si_dot_f16(n, s, x, y);
}

static void dot_q4_0_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_q4_0_q8_0(n, s, x, y);
        return;
    }
#endif
    si_dot_q4_0_q8_0(n, s, x, y);
}

static void dot_q5_0_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_q5_0_q8_0(n, s, x, y);
        return;
    }
#endif
    si_dot_q5_0_q8_0(n, s, x, y);
}

static void dot_q8_0_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_q8_0_q8_0(n, s, x, y);
        return;
    }
#endif
    si_dot_q8_0_q8_0(n, s, x, y);
}

static void dot_q2_K_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_q2_K_q8_K(n, s, x, y);
        return;
    }
#endif
    si_dot_q2_K_q8_K(n, s, x, y);
}

static void dot_q3_K_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_q3_K_q8_K(n, s, x, y);
        return;
    }
#endif
    si_dot_q3_K_q8_K(n, s, x, y);
}

static void dot_q4_K_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_q4_K_q8_K(n, s, x, y);
        return;
    }
#endif
    si_dot_q4_K_q8_K(n, s, x, y);
}

static void dot_q5_K_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_q5_K_q8_K(n, s, x, y);
        return;
    }
#endif
    si_dot_q5_K_q8_K(n, s, x, y);
}

static void dot_q6_K_dispatch(int n, float *s, const void *x, const void *y) {
#ifdef SI_X86
    if (si_kern_impl_get() == SI_KERN_AVX2) {
        si_avx2_dot_q6_K_q8_K(n, s, x, y);
        return;
    }
#endif
    si_dot_q6_K_q8_K(n, s, x, y);
}

static void dot_f32_dispatch(int n, float *s, const void *x, const void *y) {
    si_dot_f32(n, s, x, y);
}

si_vec_dot_fn si_vec_dot_for(uint32_t wtype) {
    switch (wtype) {
        case SI_T_F32:  return dot_f32_dispatch;
        case SI_T_F16:  return dot_f16_dispatch;
        case SI_T_Q4_0: return dot_q4_0_dispatch;
        case SI_T_Q5_0: return dot_q5_0_dispatch;
        case SI_T_Q8_0: return dot_q8_0_dispatch;
        case SI_T_Q2_K: return dot_q2_K_dispatch;
        case SI_T_Q3_K: return dot_q3_K_dispatch;
        case SI_T_Q4_K: return dot_q4_K_dispatch;
        case SI_T_Q5_K: return dot_q5_K_dispatch;
        case SI_T_Q6_K: return dot_q6_K_dispatch;
        default: return NULL;
    }
}
