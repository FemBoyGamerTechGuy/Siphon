/* siphon: quantized block geometry
 *
 * Byte-for-byte block layouts for the ggml quantization formats used by
 * real llama-family GGUF models. The struct field order, sizes and the
 * static asserts below mirror ggml-common.h exactly: a wrong layout is
 * silent data corruption, so every type is pinned to its on-disk size.
 *
 * Q8_K is not a file format; it is the runtime activation format the
 * K-quant dot products consume (quantized activations).
 */
#ifndef SIPHON_QUANT_H
#define SIPHON_QUANT_H

#include "siphon/common.h"

/* C++ (dev-time reference tools) understands static_assert */
#ifdef __cplusplus
#define _Static_assert(cond, msg) static_assert(cond, msg)
#endif

/* ggml half-precision storage: IEEE 754 binary16, little-endian u16. */
typedef uint16_t si_f16;

/* block size of the classic (non K) formats */
#define SI_QK4_0 32
#define SI_QK5_0 32
#define SI_QK8_0 32
/* super-block size of the K formats */
#define SI_QK_K  256

#pragma pack(push, 1)
typedef struct {
    si_f16   d;                /* delta */
    uint8_t  qs[SI_QK4_0 / 2]; /* nibbles */
} si_blk_q4_0;

typedef struct {
    si_f16   d;                /* delta */
    uint8_t  qh[4];            /* 5th bit of quants */
    uint8_t  qs[SI_QK5_0 / 2]; /* nibbles */
} si_blk_q5_0;

typedef struct {
    si_f16   d;                /* delta */
    int8_t   qs[SI_QK8_0];     /* quants */
} si_blk_q8_0;

typedef struct {
    uint8_t  scales[SI_QK_K / 16]; /* scales and mins, 4 bit each */
    uint8_t  qs[SI_QK_K / 4];      /* 2-bit quants */
    si_f16   d;                    /* super-block scale for scales */
    si_f16   dmin;                 /* super-block scale for mins */
} si_blk_q2_K;

typedef struct {
    uint8_t  hmask[SI_QK_K / 8];  /* high bit (1 bit per element) */
    uint8_t  qs[SI_QK_K / 4];     /* low 2 bits per element */
    uint8_t  scales[12];          /* 6-bit scales */
    si_f16   d;                   /* super-block scale */
} si_blk_q3_K;

typedef struct {
    si_f16   d;                    /* super-block scale for scales */
    si_f16   dmin;                 /* super-block scale for mins */
    uint8_t  scales[12];           /* 6-bit packed scales + mins */
    uint8_t  qs[SI_QK_K / 2];      /* 4-bit quants */
} si_blk_q4_K;

typedef struct {
    si_f16   d;                    /* super-block scale for scales */
    si_f16   dmin;                 /* super-block scale for mins */
    uint8_t  scales[12];           /* 6-bit packed scales + mins */
    uint8_t  qh[SI_QK_K / 8];      /* high bit */
    uint8_t  qs[SI_QK_K / 2];      /* 4-bit quants */
} si_blk_q5_K;

typedef struct {
    uint8_t  ql[SI_QK_K / 2];      /* lower 4 bits */
    uint8_t  qh[SI_QK_K / 4];      /* upper 2 bits */
    int8_t   scales[SI_QK_K / 16]; /* 8-bit scales */
    si_f16   d;                    /* super-block scale */
} si_blk_q6_K;

/* runtime-only: quantized activations for K-quant dot products */
typedef struct {
    float    d;                    /* delta */
    int8_t   qs[SI_QK_K];          /* quants */
    int16_t  bsums[SI_QK_K / 16];  /* group sums for the min correction */
} si_blk_q8_K;
#pragma pack(pop)

_Static_assert(sizeof(si_blk_q4_0) == 18,  "q4_0 block size");
_Static_assert(sizeof(si_blk_q5_0) == 22,  "q5_0 block size");
_Static_assert(sizeof(si_blk_q8_0) == 34,  "q8_0 block size");
_Static_assert(sizeof(si_blk_q2_K) == 84,  "q2_K block size");
_Static_assert(sizeof(si_blk_q3_K) == 110, "q3_K block size");
_Static_assert(sizeof(si_blk_q4_K) == 144, "q4_K block size");
_Static_assert(sizeof(si_blk_q5_K) == 176, "q5_K block size");
_Static_assert(sizeof(si_blk_q6_K) == 210, "q6_K block size");
_Static_assert(sizeof(si_blk_q8_K) == 292, "q8_K block size");

/* ggml tensor type ids (subset; numbering is the ggml enum on disk) */
enum {
    SI_T_F32   = 0,
    SI_T_F16   = 1,
    SI_T_Q4_0  = 2,
    SI_T_Q4_1  = 3,
    /* 4..5 removed legacy types */
    SI_T_Q5_0  = 6,
    SI_T_Q5_1  = 7,
    SI_T_Q8_0  = 8,
    SI_T_Q8_1  = 9,
    SI_T_Q2_K  = 10,
    SI_T_Q3_K  = 11,
    SI_T_Q4_K  = 12,
    SI_T_Q5_K  = 13,
    SI_T_Q6_K  = 14,
    SI_T_Q8_K  = 15, /* runtime only, never on disk */
};

/* per-type geometry: elements per block, bytes per block. Returns false
 * for types Siphon does not implement (IQ family etc. are milestone 6). */
bool si_quant_geometry(uint32_t type, uint32_t *blck_size, uint32_t *type_size);

/* human-readable name ("Q4_K"), "TYPE<n>" for unknown ones. */
const char *si_quant_name(uint32_t type);

/* activation format a weight type's dot product consumes (the "vec dot
 * pair"): Q8_0 blocks for the classic formats, Q8_K super-blocks for the
 * K formats, F16 for F16 weights. False for unsupported types. */
bool si_quant_vec_dot_pair(uint32_t wtype, uint32_t *pair_type);

#endif /* SIPHON_QUANT_H */
