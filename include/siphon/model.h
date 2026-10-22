/* siphon: llama-family model loading
 *
 * Loads hyperparameters, weights (zero-copy tensor views into the
 * mmap) and per-layer backend assignment for the llama and qwen2
 * architectures — the two graphs Milestone 2 executes. Norm weights
 * and biases are materialized as F32 at load (they are F32 in real
 * quantized files; F16 ones are widened losslessly), so the forward
 * pass never branches on their storage.
 */
#ifndef SIPHON_MODEL_H
#define SIPHON_MODEL_H

#include "siphon/gguf.h"
#include "siphon/tensor.h"

typedef enum {
    SI_ARCH_LLAMA = 0,   /* rope NORM: TinyLlama, Llama 1/2 */
    SI_ARCH_QWEN2 = 1,   /* rope NEOX + q/k/v biases */
} si_arch;

typedef struct {
    si_arch arch;
    uint32_t n_vocab;
    uint64_t n_ctx_train;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_rot;          /* rope dimension count == head_dim */
    uint64_t n_ff;
    float rms_eps;
    float rope_freq_base;
    float attn_scale;        /* 0 -> 1/sqrt(head_dim) */

    /* derived */
    uint32_t head_dim;
    uint32_t n_head_gqa;     /* n_head / n_head_kv */
} si_hparams;

typedef struct {
    si_tensor wq, wk, wv, wo;
    si_tensor w_gate, w_up, w_down;
    /* F32 materialized at load (NULL when absent) */
    float *attn_norm_f32;    /* [n_embd] */
    float *ffn_norm_f32;     /* [n_embd] */
    float *bq, *bk, *bv;     /* biases, [n_out] each (qwen2) */
} si_layer;

typedef struct {
    si_gguf *g;              /* owns the mmap; keep open */
    si_hparams h;
    si_arch arch;
    si_tensor tok_embd;      /* [n_embd, n_vocab] */
    si_layer *layers;
    float *out_norm_f32;     /* [n_embd] */
    si_tensor output;        /* [n_embd, n_vocab]; aliases tok_embd if tied */
    bool output_tied;
    /* per-layer backend assignment (all CPU in milestone 2) */
    si_backend **layer_be;
} si_model;

/* Load a model. The si_gguf handle is kept by the model and released by
 * si_model_free. Returns SI_OK or negative error. */
int si_model_load(const char *path, si_model **out);
void si_model_free(si_model *m);

/* Attention scale actually used (h.attn_scale==0 -> 1/sqrt(head_dim)). */
float si_model_kq_scale(const si_model *m);

#endif /* SIPHON_MODEL_H */
