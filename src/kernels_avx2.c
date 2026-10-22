/* siphon: AVX2+FMA+F16C kernels (see include/siphon/kernels.h)
 *
 * Every function here is transcribed structure-for-structure from ggml's
 * x86 kernels: the lane grouping, maddubs/madd pairing, scale shuffle
 * tables, f32 FMA accumulation order and horizontal-sum sequences match
 * the reference exactly, so the results are bit-identical, not merely
 * numerically close. Each function carries a target attribute, so one
 * binary runs on any x86-64 CPU with runtime CPUID dispatch.
 *
 * Dev-time reference: ggml arch/x86/quants.c + vec.cpp (never linked,
 * never shipped — transcription only, verified by kdiff at 0 ulp).
 */
#include "siphon/kernels.h"

#if defined(__x86_64__) || defined(__i386__)
#define SI_X86 1
#include <immintrin.h>
#endif

#ifdef SI_X86

#define SI_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))

#define SI_MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

/* ---- shared helpers (ggml arch/x86/quants.c) ------------------------ */

SI_TARGET_AVX2
static inline float si_hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

/* spread 32 bits to 32 bytes { 0x00, 0xFF } */
SI_TARGET_AVX2
static inline __m256i si_bytes_from_bits_32(const uint8_t *x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303LL, 0x0202020202020202LL,
            0x0101010101010101LL, 0x0000000000000000LL);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1LL));
}

/* unpack 32 4-bit fields into 32 bytes in [0..15] */
SI_TARGET_AVX2
static inline __m256i si_bytes_from_nibbles_32(const uint8_t *rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = SI_MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

SI_TARGET_AVX2
static inline __m256 si_sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

SI_TARGET_AVX2
static inline __m256 si_mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return si_sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}

/* scale shuffles for the K-quant dots */
SI_TARGET_AVX2
static inline __m256i si_get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

SI_TARGET_AVX2
static inline __m256i si_get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

SI_TARGET_AVX2
static inline __m128i si_get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i *)k_shuffle + i);
}

/* ---- dot products ---------------------------------------------------- */

SI_TARGET_AVX2
void si_avx2_dot_f16(int n, float *s, const si_f16 *x, const si_f16 *y) {
    /* GGML_F16_DOT: EPR 8, ARR 4, STEP 32; FMA into 4 accumulators,
     * tree-reduce, leftovers into the double sum. */
    double sumf = 0.0;
    const int np = n & ~31;
    __m256 sum[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                      _mm256_setzero_ps(), _mm256_setzero_ps() };
    __m256 ax[4], ay[4];
    for (int i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; j++) {
            ax[j] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + j * 8)));
            ay[j] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(y + i + j * 8)));
            sum[j] = _mm256_fmadd_ps(ax[j], ay[j], sum[j]);
        }
    }
    /* GGML_F32x8_REDUCE: x[0]+=x[2]; x[1]+=x[3]; x[0]+=x[1]; then
     * t0 = lo128+hi128; t1 = hadd(t0,t0); res = hadd(t1,t1). */
    sum[0] = _mm256_add_ps(sum[0], sum[2]);
    sum[1] = _mm256_add_ps(sum[1], sum[3]);
    sum[0] = _mm256_add_ps(sum[0], sum[1]);
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sum[0]),
                                 _mm256_extractf128_ps(sum[0], 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);
    sumf += (double)_mm_cvtss_f32(_mm_hadd_ps(t1, t1));

    for (int i = np; i < n; ++i) {
        sumf += (double)(si_f16_to_f32(x[i]) * si_f16_to_f32(y[i]));
    }
    *s = (float)sumf;
}

SI_TARGET_AVX2
void si_avx2_dot_q4_0_q8_0(int n, float *s, const si_blk_q4_0 *x, const si_blk_q8_0 *y) {
    const int nb = n / SI_QK8_0;
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(
            si_f16_to_f32(x[ib].d) * si_f16_to_f32(y[ib].d));
        __m256i qx = si_bytes_from_nibbles_32(x[ib].qs);
        const __m256i off = _mm256_set1_epi8(8);
        qx = _mm256_sub_epi8(qx, off);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256 q = si_mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    *s = si_hsum_float_8(acc);
}

SI_TARGET_AVX2
void si_avx2_dot_q5_0_q8_0(int n, float *s, const si_blk_q5_0 *x, const si_blk_q8_0 *y) {
    const int nb = n / SI_QK8_0;
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(
            si_f16_to_f32(x[ib].d) * si_f16_to_f32(y[ib].d));
        __m256i qx = si_bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = si_bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
        qx = _mm256_or_si256(qx, bxhi);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256 q = si_mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    *s = si_hsum_float_8(acc);
}

SI_TARGET_AVX2
void si_avx2_dot_q8_0_q8_0(int n, float *s, const si_blk_q8_0 *x, const si_blk_q8_0 *y) {
    const int nb = n / SI_QK8_0;
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(
            si_f16_to_f32(x[ib].d) * si_f16_to_f32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256 q = si_mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    *s = si_hsum_float_8(acc);
}

SI_TARGET_AVX2
void si_avx2_dot_q2_K_q8_K(int n, float *s, const si_blk_q2_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * si_f16_to_f32(x[i].d);
        const float dmin = -y[i].d * si_f16_to_f32(x[i].dmin);

        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i *)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *)y[i].bsums));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {
            SI_MM256_SET_M128I(l_scales, l_scales),
            SI_MM256_SET_M128I(h_scales, h_scales) };

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < SI_QK_K / 128; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i *)q2); q2 += 32;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], si_get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], si_get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], si_get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], si_get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = si_hsum_float_8(acc);
}

SI_TARGET_AVX2
void si_avx2_dot_q3_K_q8_K(int n, float *s, const si_blk_q3_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * si_f16_to_f32(x[i].d);

        const uint8_t *q3 = x[i].qs;
        const int8_t *q8 = y[i].qs;

        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                (int)(((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)),
                (int)(((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)),
                (int)((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)),
                (int)((aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4)));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {
            SI_MM256_SET_M128I(l_scales, l_scales),
            SI_MM256_SET_M128I(h_scales, h_scales) };

        const __m256i hbits = _mm256_loadu_si256((const __m256i *)x[i].hmask);

        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;
        int is = 0;

        for (int j = 0; j < SI_QK_K / 128; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i *)q3); q3 += 32;

            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], si_get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], si_get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], si_get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], si_get_scale_shuffle_q3k(is + 3)), p16_3);

            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = si_hsum_float_8(acc);
}

SI_TARGET_AVX2
void si_avx2_dot_q4_K_q8_K(int n, float *s, const si_blk_q4_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    uint32_t utmp[4];

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * si_f16_to_f32(x[i].d);
        const float dmin = -y[i].d * si_f16_to_f32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t *q4 = x[i].qs;
        const int8_t *q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = SI_MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < SI_QK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, si_get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, si_get_scale_shuffle_k4(2 * j + 1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i *)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = si_hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

SI_TARGET_AVX2
void si_avx2_dot_q5_K_q8_K(int n, float *s, const si_blk_q5_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t *q5 = x[i].qs;
        const int8_t *q8 = y[i].qs;

        const float d = y[i].d * si_f16_to_f32(x[i].d);
        const float dmin = -y[i].d * si_f16_to_f32(x[i].dmin);

        uint32_t utmp[4];
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = SI_MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;

        for (int j = 0; j < SI_QK_K / 64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, si_get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, si_get_scale_shuffle_k4(2 * j + 1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i *)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = si_hsum_float_8(acc) + summs;
}

SI_TARGET_AVX2
void si_avx2_dot_q6_K_q8_K(int n, float *s, const si_blk_q6_K *x, const si_blk_q8_K *y) {
    const int nb = n / SI_QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * si_f16_to_f32(x[i].d);

        const uint8_t *q4 = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i *)x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < SI_QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, si_get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, si_get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, si_get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, si_get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = si_hsum_float_8(acc);
}

/* ---- activation quantization ---------------------------------------- */

SI_TARGET_AVX2
void si_avx2_quant_row_q8_0(const float *x, si_blk_q8_0 *y, int n) {
    const int nb = n / SI_QK8_0;
    for (int i = 0; i < nb; i++) {
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                                 _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        const float d = maxScalar / 127.f;
        y[i].d = si_f32_to_f16(d);
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);

        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);

        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
    }
}

/* ---- exp polynomial (vector form, used by tests to cross-check the
 * scalar transcription in kernels.c) ---------------------------------- */

SI_TARGET_AVX2
__m256 si_avx2_v_expf(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(
        _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                      _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)),
            u,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
        return _mm256_fmadd_ps(j, k, k);
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000u));
    const __m256 s1 = _mm256_castsi256_ps(
        _mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                      _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(_mm256_castsi256_ps(c),
                              _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

/* declared in kernels.c/tests via a memory-based interface so callers
 * compiled without AVX flags do not change ABI. Evaluates the ggml
 * vector exp on 8 floats from x into y; used to prove the scalar
 * transcription is lane-identical to the vector form. */
void si_avx2_v_expf_pub8(const float *x, float *y);

SI_TARGET_AVX2
void si_avx2_v_expf_pub8(const float *x, float *y) {
    const __m256 v = si_avx2_v_expf(_mm256_loadu_ps(x));
    _mm256_storeu_ps(y, v);
}

#else /* !SI_X86 */

/* non-x86 ports: the dispatch in kernels.c falls back to scalar and these
 * symbols are referenced only by x86 tests. */

#endif /* SI_X86 */
