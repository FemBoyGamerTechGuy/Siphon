/* siphon: sampling
 *
 * Greedy (temperature 0) picks the first argmax, byte-identical to the
 * reference's greedy rule. Stochastic sampling applies temperature,
 * top-k and top-p (nucleus) in that order over the softmax of the
 * filtered logits; the RNG is a portable xorshift64* so runs are
 * reproducible from --seed.
 */
#ifndef SIPHON_SAMPLER_H
#define SIPHON_SAMPLER_H

#include "siphon/common.h"

typedef struct {
    float temp;      /* 0 = greedy */
    int32_t top_k;   /* 0 = off */
    float top_p;     /* 1.0 = off */
    uint64_t seed;   /* initial seed */
    uint64_t rng;    /* evolving state; initialized from seed on first use */
} si_sampler;

/* first argmax (reference greedy rule: first occurrence wins) */
uint32_t si_sample_greedy(const float *logits, int32_t n_vocab);

/* full sampler; returns a token id (advances the internal RNG state) */
uint32_t si_sample(si_sampler *s, const float *logits, int32_t n_vocab);

#endif /* SIPHON_SAMPLER_H */
