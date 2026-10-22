/* siphon: sampling (see include/siphon/sampler.h) */
#include "siphon/sampler.h"

#include <math.h>

uint32_t si_sample_greedy(const float *logits, int32_t n_vocab) {
    int32_t best = 0;
    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return (uint32_t)best;
}

static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

typedef struct {
    float p;
    int32_t idx;
} cand_t;

static int cmp_cand(const void *a, const void *b) {
    const cand_t *ca = a, *cb = b;
    if (ca->p > cb->p) return -1;
    if (ca->p < cb->p) return 1;
    return 0;
}

uint32_t si_sample(si_sampler *s, const float *logits, int32_t n_vocab) {
    if (s->temp <= 0.0f || (s->top_k == 1 && s->top_p >= 1.0f)) {
        return si_sample_greedy(logits, n_vocab);
    }
    if (s->rng == 0) {
        s->rng = s->seed ? s->seed : 0x9E3779B97F4A7C15ULL;
    }

    const float temp = s->temp > 0.0f ? s->temp : 1.0f;

    /* collect candidates above the max - temp, softmax over all */
    float max = logits[0];
    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > max) max = logits[i];
    }
    cand_t *c = si_xmalloc(sizeof(cand_t) * (size_t)n_vocab);
    double total = 0.0;
    for (int32_t i = 0; i < n_vocab; i++) {
        c[i].p = expf((logits[i] - max) / temp);
        c[i].idx = i;
        total += c[i].p;
    }
    for (int32_t i = 0; i < n_vocab; i++) {
        c[i].p = (float)(c[i].p / total);
    }

    int32_t n = n_vocab;

    if (s->top_k > 0 && s->top_k < n) {
        qsort(c, (size_t)n, sizeof *c, cmp_cand);
        n = s->top_k;
    } else {
        qsort(c, (size_t)n, sizeof *c, cmp_cand);
    }

    if (s->top_p < 1.0f) {
        double cum = 0.0;
        int32_t last = n;
        for (int32_t i = 0; i < n; i++) {
            cum += c[i].p;
            if (cum >= s->top_p) {
                last = i + 1;
                break;
            }
        }
        n = last < 1 ? 1 : last;
    }

    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) sum += c[i].p;
    double r = (double)(rng_next(&s->rng) >> 11) / 9007199254740992.0;
    double pick = r * sum;
    double acc = 0.0;
    int32_t chosen = c[n - 1].idx;
    for (int32_t i = 0; i < n; i++) {
        acc += c[i].p;
        if (pick <= acc) {
            chosen = c[i].idx;
            break;
        }
    }
    free(c);
    return (uint32_t)chosen;
}
