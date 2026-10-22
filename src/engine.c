/* siphon: inference engine — llama/qwen2 forward (see include/siphon/engine.h) */
#include "siphon/engine.h"
#include "siphon/kernels.h"

#include <math.h>

static si_vec_dot_fn g_dot_f16;

/* ---- rmsnorm + weight (the reference's fused form) -------------------- */

static void rms_norm_mul(float *y, const float *x, const float *w,
                         int n, float eps) {
    const float scale = si_rms_scale(x, n, eps);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * scale * w[i];
    }
}

/* ---- rope ---------------------------------------------------------------
 * angle table per position exactly like the reference: theta starts at
 * the position and multiplies by theta_scale per pair; cos/sin via the
 * C library (same libm as the reference build).
 * ----------------------------------------------------------------------- */

static void rope_table(si_engine *e, int64_t pos) {
    float theta = (float)pos;
    const int np = (int)e->m->h.n_rot / 2;
    for (int j = 0; j < np; j++) {
        e->rope_cos[j] = cosf(theta);
        e->rope_sin[j] = sinf(theta);
        theta *= e->theta_scale;
    }
}

/* NORM (interleaved pairs) or NEOX (half-split pairs) in place. */
static void rope_apply(si_engine *e, float *v, int head_dim) {
    const int n_rot = (int)e->m->h.n_rot;
    const int np = n_rot / 2;
    if (e->m->arch == SI_ARCH_LLAMA) {
        for (int j = 0; j < np; j++) {
            const float x0 = v[2 * j + 0];
            const float x1 = v[2 * j + 1];
            const float cs = e->rope_cos[j];
            const float sn = e->rope_sin[j];
            v[2 * j + 0] = x0 * cs - x1 * sn;
            v[2 * j + 1] = x0 * sn + x1 * cs;
        }
    } else { /* NEOX */
        const int half = head_dim / 2;
        for (int j = 0; j < np; j++) {
            const float x0 = v[j];
            const float x1 = v[j + half];
            const float cs = e->rope_cos[j];
            const float sn = e->rope_sin[j];
            v[j] = x0 * cs - x1 * sn;
            v[j + half] = x0 * sn + x1 * cs;
        }
    }
    /* channels beyond n_rot pass through unchanged (not used by the
     * milestone-2 models, but keep the semantics correct) */
}

/* ---- attention task ------------------------------------------------------
 * One query head end-to-end inside one lane: score dots, softmax, value
 * contraction. Lane count therefore cannot change any bit.
 * ------------------------------------------------------------------------ */

typedef struct {
    si_engine *e;
    uint32_t il;
    int64_t pos;   /* current position (scores span 0..pos) */
    const float *q;         /* [n_embd_q] */
    float *out;             /* [n_embd] */
} attn_task;

static void attn_head(si_engine *e, uint32_t il, int64_t pos, int lane,
                      const float *q, float *out, int h) {
    const si_hparams *hp = &e->m->h;
    const int hd = (int)hp->head_dim;
    const int h_kv = h / (int)hp->n_head_gqa;
    const int n_score = (int)pos + 1;

    /* per-lane scratch so concurrent heads never share buffers */
    float *sc = e->scores + (size_t)lane * (size_t)e->n_ctx;
    float *pr = e->probs + (size_t)lane * (size_t)e->n_ctx;
    si_f16 *pr16 = e->probs16 + (size_t)lane * (size_t)e->n_ctx;

    /* base of this layer+head in the caches */
    si_f16 *kh = e->k_cache +
        ((size_t)il * hp->n_head_kv + (size_t)h_kv) *
        (size_t)e->n_ctx * hp->head_dim;
    si_f16 *vh = e->v_cache +
        ((size_t)il * hp->n_head_kv + (size_t)h_kv) *
        (size_t)hp->head_dim * (size_t)e->n_ctx;

    /* Q slice -> F16 (element-wise RNE, identical bits regardless of
     * where the row boundaries fall) */
    si_f16 *q16 = e->q16 + (size_t)h * hd;
    const float *qh = q + (size_t)h * hd;
    for (int d = 0; d < hd; d++) {
        q16[d] = si_f32_to_f16(qh[d]);
    }

    /* scores: dispatched f16 dot per cached position */
    for (int j = 0; j < n_score; j++) {
        g_dot_f16(hd, &sc[j], kh + (size_t)j * hd, q16);
    }

    /* scale, max */
    const float ks = e->kq_scale;
    for (int j = 0; j < n_score; j++) {
        sc[j] = sc[j] * ks;
    }
    float max = -INFINITY;
    for (int j = 0; j < n_score; j++) {
        if (sc[j] > max) max = sc[j];
    }

    /* softmax with the reference's accumulation order */
    si_softmax(pr, sc, n_score, max);

    /* probs -> F16 (the reference quantizes the softmax row before the
     * value contraction) */
    for (int j = 0; j < n_score; j++) {
        pr16[j] = si_f32_to_f16(pr[j]);
    }

    /* value contraction through the same f16 dot over transposed V */
    float *oh = out + (size_t)h * hd;
    for (int c = 0; c < hd; c++) {
        g_dot_f16(n_score, &oh[c], vh + (size_t)c * e->n_ctx, pr16);
    }
}

static void attn_task_fn(void *ctxp, int lane, int lanes) {
    attn_task *t = ctxp;
    const int n_head = (int)t->e->m->h.n_head;
    const int per = (n_head + lanes - 1) / lanes;
    const int h0 = lane * per;
    const int h1 = h0 + per < n_head ? h0 + per : n_head;
    for (int h = h0; h < h1; h++) {
        attn_head(t->e, t->il, t->pos, lane, t->q, t->out, h);
    }
}

/* ---- swiglu ------------------------------------------------------------- */

static void swiglu_row(float *y, const float *x, const float *g, int n) {
    /* reference order: 8-wide groups through the exp-poly silu, scalar
     * tail through expf; then multiply by g elementwise */
    int i = 0;
    for (; i + 7 < n; i += 8) {
        y[i + 0] = si_silu(x[i + 0]) * g[i + 0];
        y[i + 1] = si_silu(x[i + 1]) * g[i + 1];
        y[i + 2] = si_silu(x[i + 2]) * g[i + 2];
        y[i + 3] = si_silu(x[i + 3]) * g[i + 3];
        y[i + 4] = si_silu(x[i + 4]) * g[i + 4];
        y[i + 5] = si_silu(x[i + 5]) * g[i + 5];
        y[i + 6] = si_silu(x[i + 6]) * g[i + 6];
        y[i + 7] = si_silu(x[i + 7]) * g[i + 7];
    }
    for (; i < n; i++) {
        y[i] = x[i] / (1.0f + expf(0.0f - x[i])) * g[i];
    }
}

/* ---- engine lifecycle ---------------------------------------------------- */

int si_engine_create(si_model *m, int64_t n_ctx, int n_threads,
                     si_engine **outp) {
    if (n_ctx < 1) n_ctx = (int64_t)m->h.n_ctx_train ? m->h.n_ctx_train : 512;
    if (n_ctx > (int64_t)m->h.n_ctx_train) {
        si_log_warn("context %lld exceeds training context %llu (rope extrapolation)",
                    (long long)n_ctx, (unsigned long long)m->h.n_ctx_train);
    }
    si_engine *e = si_xcalloc(1, sizeof *e);
    e->m = m;
    e->tp = si_tp_create(n_threads);
    e->n_ctx = n_ctx;
    e->n_pos = 0;
    e->kq_scale = si_model_kq_scale(m);
    e->theta_scale = powf(m->h.rope_freq_base, -2.0f / (float)m->h.n_rot);
    g_dot_f16 = si_vec_dot_for(SI_T_F16);

    const si_hparams *h = &m->h;
    e->k_cache = si_xmalloc((size_t)h->n_layer * h->n_head_kv *
                            (size_t)n_ctx * h->head_dim * sizeof(si_f16));
    e->v_cache = si_xmalloc((size_t)h->n_layer * h->n_head_kv *
                            (size_t)h->head_dim * (size_t)n_ctx * sizeof(si_f16));

    e->x  = si_xmalloc(h->n_embd * sizeof(float));
    e->xb = si_xmalloc(h->n_embd * sizeof(float));
    e->q  = si_xmalloc(h->n_embd * sizeof(float));
    e->k  = si_xmalloc(h->n_head_kv * h->head_dim * sizeof(float));
    e->v  = si_xmalloc(h->n_head_kv * h->head_dim * sizeof(float));
    e->attn_out = si_xmalloc(h->n_embd * sizeof(float));
    e->gate = si_xmalloc(h->n_ff * sizeof(float));
    e->up   = si_xmalloc(h->n_ff * sizeof(float));
    e->ffbuf = si_xmalloc(h->n_embd * sizeof(float));
    e->logits = si_xmalloc((size_t)h->n_vocab * sizeof(float));
    e->rope_cos = si_xmalloc(h->n_rot / 2 * sizeof(float));
    e->rope_sin = si_xmalloc(h->n_rot / 2 * sizeof(float));
    const int lanes = si_tp_nlanes(e->tp);
    e->scores = si_xmalloc((size_t)lanes * (size_t)n_ctx * sizeof(float));
    e->probs  = si_xmalloc((size_t)lanes * (size_t)n_ctx * sizeof(float));
    e->q16 = si_xmalloc(h->n_embd * sizeof(si_f16));
    e->probs16 = si_xmalloc((size_t)lanes * (size_t)n_ctx * sizeof(si_f16));

    *outp = e;
    return SI_OK;
}

void si_engine_reset(si_engine *e) {
    e->n_pos = 0;
}

void si_engine_free(si_engine *e) {
    if (!e) return;
    si_tp_free(e->tp);
    free(e->k_cache);
    free(e->v_cache);
    free(e->x); free(e->xb);
    free(e->q); free(e->k); free(e->v);
    free(e->attn_out);
    free(e->gate); free(e->up); free(e->ffbuf);
    free(e->logits);
    free(e->rope_cos); free(e->rope_sin);
    free(e->scores); free(e->probs);
    free(e->q16); free(e->probs16);
    free(e);
}

/* ---- forward -------------------------------------------------------------- */

const float *si_engine_forward(si_engine *e, uint32_t token) {
    si_model *m = e->m;
    const si_hparams *h = &m->h;
    const int64_t pos = e->n_pos;
    if (pos >= e->n_ctx) {
        si_set_err("context full (%lld tokens)", (long long)e->n_ctx);
        return NULL;
    }
    const int n_embd = (int)h->n_embd;
    const int hd = (int)h->head_dim;
    const int n_head_kv = (int)h->n_head_kv;

    /* embedding row (dequantized through the backend vtable) */
    {
        si_backend *be0 = si_backend_cpu();
        be0->row_to_f32(be0, &m->tok_embd, token, e->x);
    }

    rope_table(e, pos);

    for (uint32_t il = 0; il < h->n_layer; il++) {
        si_layer *l = &m->layers[il];
        si_backend *be = m->layer_be[il];

        /* attention block */
        rms_norm_mul(e->xb, e->x, l->attn_norm_f32, n_embd, h->rms_eps);

        be->gemv(be, &l->wq, e->xb, e->q, (int64_t)h->n_embd, e->tp);
        be->gemv(be, &l->wk, e->xb, e->k, (int64_t)(n_head_kv * hd), e->tp);
        be->gemv(be, &l->wv, e->xb, e->v, (int64_t)(n_head_kv * hd), e->tp);
        if (l->bq) {
            for (int i = 0; i < n_embd; i++) e->q[i] = e->q[i] + l->bq[i];
        }
        if (l->bk) {
            for (int i = 0; i < n_head_kv * hd; i++) e->k[i] = e->k[i] + l->bk[i];
        }
        if (l->bv) {
            for (int i = 0; i < n_head_kv * hd; i++) e->v[i] = e->v[i] + l->bv[i];
        }

        /* rope on every head of q and k */
        for (uint32_t hh = 0; hh < h->n_head; hh++) {
            rope_apply(e, e->q + hh * hd, hd);
        }
        for (uint32_t hh = 0; hh < h->n_head_kv; hh++) {
            rope_apply(e, e->k + hh * hd, hd);
        }

        /* cache writes: K position-major, V transposed */
        si_f16 *kbase = e->k_cache +
            ((size_t)il * n_head_kv) * (size_t)e->n_ctx * hd +
            (size_t)pos * hd;
        for (int hh = 0; hh < n_head_kv; hh++) {
            si_f16 *dst = kbase + (size_t)hh * (size_t)e->n_ctx * hd;
            const float *src = e->k + (size_t)hh * hd;
            for (int d = 0; d < hd; d++) dst[d] = si_f32_to_f16(src[d]);
        }
        si_f16 *vbase = e->v_cache +
            ((size_t)il * n_head_kv) * (size_t)hd * (size_t)e->n_ctx +
            (size_t)pos;
        for (int hh = 0; hh < n_head_kv; hh++) {
            si_f16 *dst = vbase + (size_t)hh * (size_t)hd * (size_t)e->n_ctx;
            const float *src = e->v + (size_t)hh * hd;
            for (int d = 0; d < hd; d++) dst[(size_t)d * (size_t)e->n_ctx] =
                si_f32_to_f16(src[d]);
        }

        /* attention over heads (parallelized by head, lane-invariant) */
        attn_task at = { .e = e, .il = il, .pos = pos, .q = e->q, .out = e->attn_out };
        si_tp_run(e->tp, attn_task_fn, &at);

        /* output projection + residual */
        be->gemv(be, &l->wo, e->attn_out, e->ffbuf, (int64_t)n_embd, e->tp);
        for (int i = 0; i < n_embd; i++) e->x[i] = e->x[i] + e->ffbuf[i];

        /* ffn block */
        rms_norm_mul(e->xb, e->x, l->ffn_norm_f32, n_embd, h->rms_eps);
        be->gemv(be, &l->w_gate, e->xb, e->gate, (int64_t)h->n_ff, e->tp);
        be->gemv(be, &l->w_up, e->xb, e->up, (int64_t)h->n_ff, e->tp);
        swiglu_row(e->gate, e->gate, e->up, (int)h->n_ff);
        be->gemv(be, &l->w_down, e->gate, e->ffbuf, (int64_t)n_embd, e->tp);
        for (int i = 0; i < n_embd; i++) e->x[i] = e->x[i] + e->ffbuf[i];

        if (e->layer_cb) {
            e->layer_cb(e, (int)il, e->x);
        }
    }

    /* final norm + head */
    rms_norm_mul(e->xb, e->x, m->out_norm_f32, n_embd, h->rms_eps);
    if (e->layer_cb) {
        e->layer_cb(e, (int)h->n_layer, e->xb);
    }
    si_backend *be = si_backend_cpu();
    be->gemv(be, &m->output, e->xb, e->logits, (int64_t)h->n_vocab, e->tp);

    e->n_pos = pos + 1;
    return e->logits;
}
