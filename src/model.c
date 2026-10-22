/* siphon: llama-family model loading (see include/siphon/model.h) */
#include "siphon/model.h"
#include "siphon/kernels.h"

#include <math.h>

static int find_tensor(si_gguf *g, const char *name, si_tensor *out) {
    const si_gguf_tensor *t = si_gguf_tensor_find(g, name);
    if (!t) {
        si_set_err("missing tensor %s", name);
        return SI_ENOENT;
    }
    return si_tensor_from_gguf(g, t, si_backend_cpu(), out);
}

/* build a zero-copy view over a small 1-D tensor (norm/bias); the
 * view is absent (data == NULL) if the tensor is not in the file */
static void find_wview(si_gguf *g, const char *name, uint64_t n,
                        si_wview *out) {
    const si_gguf_tensor *t = si_gguf_tensor_find(g, name);
    out->data = t ? t->data : NULL;
    out->type = t ? t->ggml_type : 0;
    out->n = n;
}

int si_model_load(const char *path, si_model **outp) {
    si_gguf *g = NULL;
    int rc = si_gguf_open(path, &g);
    if (rc != SI_OK) return rc;

    si_model *m = si_xcalloc(1, sizeof *m);
    m->g = g;

    si_gguf_str arch_s = si_gguf_get_str(g, "general.architecture");
    char arch[32] = {0};
    if (arch_s.ptr && arch_s.len < sizeof arch) {
        memcpy(arch, arch_s.ptr, arch_s.len);
    }

    if (strcmp(arch, "llama") == 0) {
        m->arch = SI_ARCH_LLAMA;
    } else if (strcmp(arch, "qwen2") == 0 || strcmp(arch, "qwen3") == 0) {
        m->arch = SI_ARCH_QWEN2;
    } else {
        si_set_err("architecture '%s' not supported yet (milestone 2: llama, qwen2)",
                   arch[0] ? arch : "?");
        si_model_free(m);
        return SI_ENOTSUP;
    }

    /* hyperparameters: {arch}.* keys */
    char k[96];
    si_hparams *h = &m->h;
    h->n_layer   = (uint32_t)si_gguf_get_u64(g, (snprintf(k, sizeof k, "%s.block_count", arch), k), 0);
    h->n_ctx_train = si_gguf_get_u64(g, (snprintf(k, sizeof k, "%s.context_length", arch), k), 0);
    h->n_embd    = (uint32_t)si_gguf_get_u64(g, (snprintf(k, sizeof k, "%s.embedding_length", arch), k), 0);
    h->n_head    = (uint32_t)si_gguf_get_u64(g, (snprintf(k, sizeof k, "%s.attention.head_count", arch), k), 0);
    h->n_head_kv = (uint32_t)si_gguf_get_u64(g, (snprintf(k, sizeof k, "%s.attention.head_count_kv", arch), k), h->n_head);
    h->n_ff      = si_gguf_get_u64(g, (snprintf(k, sizeof k, "%s.feed_forward_length", arch), k), 0);
    h->rms_eps   = (float)si_gguf_get_f64(g, (snprintf(k, sizeof k, "%s.attention.layer_norm_rms_epsilon", arch), k), 1e-5);
    h->rope_freq_base = (float)si_gguf_get_f64(g, (snprintf(k, sizeof k, "%s.rope.freq_base", arch), k), 10000.0);
    h->attn_scale = (float)si_gguf_get_f64(g, (snprintf(k, sizeof k, "%s.attention.scale", arch), k), 0.0);
    const uint32_t key_len = (uint32_t)si_gguf_get_u64(g, (snprintf(k, sizeof k, "%s.attention.key_length", arch), k), 0);
    const uint32_t val_len = (uint32_t)si_gguf_get_u64(g, (snprintf(k, sizeof k, "%s.attention.value_length", arch), k), 0);

    if (!h->n_layer || !h->n_embd || !h->n_head || !h->n_ff) {
        si_set_err("missing required hyperparameters");
        si_model_free(m);
        return SI_EFORMAT;
    }
    if (h->n_head % h->n_head_kv != 0) {
        si_set_err("head_count %u not divisible by head_count_kv %u",
                   h->n_head, h->n_head_kv);
        si_model_free(m);
        return SI_EFORMAT;
    }
    h->head_dim = key_len ? key_len : h->n_embd / h->n_head;
    if (val_len && val_len != h->head_dim) {
        si_set_err("value_length != key_length not supported");
        si_model_free(m);
        return SI_ENOTSUP;
    }
    h->n_rot = h->head_dim;
    h->n_head_gqa = h->n_head / h->n_head_kv;

    rc = find_tensor(g, "token_embd.weight", &m->tok_embd);
    if (rc != SI_OK) { si_model_free(m); return rc; }
    h->n_vocab = (uint32_t)m->tok_embd.ne[1];

    if (si_gguf_tensor_find(g, "output.weight")) {
        rc = find_tensor(g, "output.weight", &m->output);
        if (rc != SI_OK) { si_model_free(m); return rc; }
        m->output_tied = false;
    } else {
        m->output = m->tok_embd;   /* tied embeddings */
        m->output_tied = true;
    }
    if (m->output.ne[1] != h->n_vocab) {
        si_set_err("output vocab %llu != embedding vocab %u",
                   (unsigned long long)m->output.ne[1], h->n_vocab);
        si_model_free(m);
        return SI_EFORMAT;
    }

    m->layers = si_xcalloc(h->n_layer, sizeof *m->layers);
    m->layer_be = si_xcalloc(h->n_layer, sizeof *m->layer_be);

    find_wview(g, "output_norm.weight", h->n_embd, &m->out_norm);
    if (!m->out_norm.data) {
        si_set_err("missing output_norm.weight");
        si_model_free(m);
        return SI_ENOENT;
    }

    /* optional per-dim rope frequency factors (rope_freqs.weight):
     * llama3-style conversions bake the low/high frequency correction
     * into the file as [n_rot/2] factors; the engine divides each
     * pair's angle by them exactly like the reference (theta/ff on the
     * accumulated angle). Absent in llama1/2, qwen, mistral files. */
    find_wview(g, "rope_freqs.weight", h->n_rot / 2, &m->rope_ffs);

    for (uint32_t il = 0; il < h->n_layer; il++) {
        si_layer *l = &m->layers[il];
        char name[128];

        snprintf(name, sizeof name, "blk.%u.attn_norm.weight", il);
        find_wview(g, name, h->n_embd, &l->attn_norm);
        if (!l->attn_norm.data) {
            si_set_err("missing %s", name);
            si_model_free(m);
            return SI_ENOENT;
        }
        snprintf(name, sizeof name, "blk.%u.ffn_norm.weight", il);
        find_wview(g, name, h->n_embd, &l->ffn_norm);
        if (!l->ffn_norm.data) {
            si_set_err("missing %s", name);
            si_model_free(m);
            return SI_ENOENT;
        }

        snprintf(name, sizeof name, "blk.%u.attn_q.weight", il);
        rc = find_tensor(g, name, &l->wq);
        if (rc != SI_OK) { si_model_free(m); return rc; }
        snprintf(name, sizeof name, "blk.%u.attn_k.weight", il);
        rc = find_tensor(g, name, &l->wk);
        if (rc != SI_OK) { si_model_free(m); return rc; }
        snprintf(name, sizeof name, "blk.%u.attn_v.weight", il);
        rc = find_tensor(g, name, &l->wv);
        if (rc != SI_OK) { si_model_free(m); return rc; }
        snprintf(name, sizeof name, "blk.%u.attn_output.weight", il);
        rc = find_tensor(g, name, &l->wo);
        if (rc != SI_OK) { si_model_free(m); return rc; }

        snprintf(name, sizeof name, "blk.%u.attn_q.bias", il);
        find_wview(g, name, l->wq.ne[0], &l->bq);
        snprintf(name, sizeof name, "blk.%u.attn_k.bias", il);
        find_wview(g, name, l->wk.ne[0], &l->bk);
        snprintf(name, sizeof name, "blk.%u.attn_v.bias", il);
        find_wview(g, name, l->wv.ne[0], &l->bv);

        snprintf(name, sizeof name, "blk.%u.ffn_gate.weight", il);
        rc = find_tensor(g, name, &l->w_gate);
        if (rc != SI_OK) { si_model_free(m); return rc; }
        snprintf(name, sizeof name, "blk.%u.ffn_up.weight", il);
        rc = find_tensor(g, name, &l->w_up);
        if (rc != SI_OK) { si_model_free(m); return rc; }
        snprintf(name, sizeof name, "blk.%u.ffn_down.weight", il);
        rc = find_tensor(g, name, &l->w_down);
        if (rc != SI_OK) { si_model_free(m); return rc; }

        m->layer_be[il] = si_backend_cpu();
    }

    *outp = m;
    return SI_OK;
}

void si_model_free(si_model *m) {
    if (!m) return;
    free(m->layers);
    free(m->layer_be);
    si_gguf_close(m->g);
    free(m);
}

float si_model_kq_scale(const si_model *m) {
    if (m->h.attn_scale != 0.0f) return m->h.attn_scale;
    return 1.0f / sqrtf((float)m->h.head_dim);
}
