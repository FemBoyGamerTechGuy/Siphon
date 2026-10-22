/* siphon: llama-family model loading
 *
 * Loads hyperparameters, weights (zero-copy tensor views into the
 * mmap) and per-layer backend assignment for the llama and qwen2
 * architectures — the two graphs Milestone 2 executes. Every weight,
 * including 1-D norm/bias vectors, is a view into the mmap: nothing is
 * copied to the heap at load. The engine materializes non-F32 views
 * into activation scratch on use (deterministic, so results are
 * bit-identical to materializing once at load).
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

/* A small 1-D weight (norm/bias) kept zero-copy as a view into the
 * mmap'd GGUF. data == NULL means the tensor is absent in the file.
 * F32 views are consumed directly; other types are dequantized into
 * engine scratch on use. */
typedef struct {
    const void *data;   /* mmap pointer (NULL if absent) */
    uint32_t type;      /* ggml type id (SI_T_*) */
    uint64_t n;         /* element count */
} si_wview;

typedef struct {
    si_tensor wq, wk, wv, wo;
    si_tensor w_gate, w_up, w_down;
    /* zero-copy views (data == NULL when absent) */
    si_wview attn_norm;      /* [n_embd] */
    si_wview ffn_norm;       /* [n_embd] */
    si_wview bq, bk, bv;     /* biases, [n_out] each (qwen2) */
} si_layer;

typedef struct {
    si_gguf *g;              /* owns the mmap; keep open */
    si_hparams h;
    si_arch arch;
    si_tensor tok_embd;      /* [n_embd, n_vocab] */
    si_layer *layers;
    si_wview out_norm;       /* [n_embd] */
    si_wview rope_ffs;       /* [n_rot/2] optional rope frequency factors */
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
