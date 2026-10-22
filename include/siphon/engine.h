/* siphon: inference engine (llama + qwen2 graphs, CPU milestone)
 *
 * Executes one token at a time through the full graph:
 *
 *   x    = dequant(tok_embd[token])
 *   per layer:
 *     xb   = rmsnorm(x) * attn_norm
 *     q,k  = Wq*xb+bq, Wk*xb+bk   (roped: NORM for llama, NEOX for qwen2)
 *     v    = Wv*xb+bv
 *     K[p] = f16(k)   (position-major cache)
 *     V[:,p] = f16(v) (transposed cache: channel-major rows)
 *     per query head: scores = f16dot(K[0..p], f16(Q)) * 1/sqrt(d)
 *                     probs  = softmax(scores)
 *                     out    = f16dot(V rows, f16(probs))
 *     x   += Wo*concat(out)
 *     x   += Wdown * silu(Wgate*xb2) * (Wup*xb2)
 *   logits = Wout * rmsnorm(x) * out_norm
 *
 * Every numerical step mirrors the reference implementation's exact
 * operation order (see kernels.h), so token-serial decode reproduces
 * the reference's logits within its own FA-on/off variation band and
 * typically bit-exactly for many layers.
 */
#ifndef SIPHON_ENGINE_H
#define SIPHON_ENGINE_H

#include "siphon/model.h"

struct si_engine;

typedef struct si_engine {
    si_model *m;
    si_tp *tp;

    /* optional debug hook: called with the residual stream after each
     * completed layer (layer == n_layer means after the final norm).
     * NULL in production runs. */
    void (*layer_cb)(struct si_engine *e, int layer, const float *x);
    void *layer_cb_ctx;

    int64_t n_ctx;             /* allocated slots */
    int64_t n_pos;             /* tokens processed (next write index) */

    /* KV cache: K [n_layer][n_head_kv][n_ctx][head_dim]
     *           V [n_layer][n_head_kv][head_dim][n_ctx] (transposed) */
    si_f16 *k_cache;
    si_f16 *v_cache;

    /* activation buffers */
    float *x, *xb;             /* [n_embd] residual stream + normed */
    float *q, *k, *v;          /* [n_embd_q] [n_embd_k] [n_embd_v] */
    float *attn_out;           /* [n_embd] */
    float *gate, *up;          /* [n_ff] */
    float *ffbuf;              /* [n_embd] */
    float *logits;             /* [n_vocab] */

    /* per-step scratch */
    float *rope_cos, *rope_sin;/* [n_rot/2] */
    float *scores;             /* [n_ctx] */
    float *probs;              /* [n_ctx] */
    si_f16 *q16;               /* [n_embd_q] */
    si_f16 *probs16;           /* [n_ctx] */

    float theta_scale;         /* powf(freq_base, -2/n_rot) */
    float kq_scale;
} si_engine;

int si_engine_create(si_model *m, int64_t n_ctx, int n_threads,
                     si_engine **out);
void si_engine_free(si_engine *e);

/* Run one token at position n_pos and advance. Returns the logits row
 * for this position (engine-owned, valid until the next call). */
const float *si_engine_forward(si_engine *e, uint32_t token);

/* Reset for a fresh sequence (positions rewind, caches stay allocated). */
void si_engine_reset(si_engine *e);

#endif /* SIPHON_ENGINE_H */
