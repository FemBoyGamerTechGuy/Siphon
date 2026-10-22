/* siphon: host/CPU backend (see include/siphon/tensor.h) */
#include "siphon/tensor.h"
#include "siphon/kernels.h"

/* ---- activation conversion task ---------------------------------------
 * Converts the F32 activation row into the dot-pair format, split across
 * lanes at block granularity so the result never depends on lane count.
 * ---------------------------------------------------------------------- */

typedef struct {
    const float *x;
    void *out;
    int64_t n;            /* elements */
    uint32_t pair;        /* SI_T_F16 / SI_T_Q8_0 / SI_T_Q8_K */
} conv_task;

static void conv_task_fn(void *ctxp, int lane, int lanes) {
    conv_task *c = ctxp;
    if (c->pair == SI_T_F16) {
        /* element-wise RNE widening: any split is fine */
        si_f16 *out = c->out;
        const int64_t per = (c->n + lanes - 1) / lanes;
        const int64_t b0 = (int64_t)lane * per;
        const int64_t b1 = b0 + per < c->n ? b0 + per : c->n;
        for (int64_t i = b0; i < b1; i++) {
            out[i] = si_f32_to_f16(c->x[i]);
        }
    } else if (c->pair == SI_T_Q8_0) {
        si_blk_q8_0 *out = c->out;
        const int64_t nblk = c->n / SI_QK8_0;
        const int64_t per = (nblk + lanes - 1) / lanes;
        const int64_t b0 = (int64_t)lane * per;
        const int64_t b1 = b0 + per < nblk ? b0 + per : nblk;
        for (int64_t b = b0; b < b1; b++) {
            si_quant_row_q8_0(c->x + b * SI_QK8_0, &out[b], SI_QK8_0);
        }
    } else { /* SI_T_Q8_K */
        si_blk_q8_K *out = c->out;
        const int64_t nblk = c->n / SI_QK_K;
        const int64_t per = (nblk + lanes - 1) / lanes;
        const int64_t b0 = (int64_t)lane * per;
        const int64_t b1 = b0 + per < nblk ? b0 + per : nblk;
        for (int64_t b = b0; b < b1; b++) {
            si_quant_row_q8_K(c->x + b * SI_QK_K, &out[b], SI_QK_K);
        }
    }
}

/* ---- row dot task ------------------------------------------------------ */

typedef struct {
    const si_tensor *w;
    const void *xq;       /* converted activation */
    float *y;
    int64_t n_out;
    int64_t row_bytes;
    int64_t n_in;
    si_vec_dot_fn dot;
} dot_task;

static void dot_task_fn(void *ctxp, int lane, int lanes) {
    dot_task *d = ctxp;
    const int64_t per = (d->n_out + lanes - 1) / lanes;
    const int64_t r0 = (int64_t)lane * per;
    const int64_t r1 = r0 + per < d->n_out ? r0 + per : d->n_out;
    const char *base = d->w->data;
    for (int64_t r = r0; r < r1; r++) {
        d->dot((int)d->n_in, &d->y[r],
               base + (size_t)r * (size_t)d->row_bytes, d->xq);
    }
}

/* ---- gemv -------------------------------------------------------------- */

static size_t pair_row_bytes(uint32_t pair, int64_t n) {
    switch (pair) {
        case SI_T_F32:  return (size_t)n * 4;
        case SI_T_F16:  return (size_t)n * 2;
        case SI_T_Q8_0: return (size_t)(n / SI_QK8_0) * sizeof(si_blk_q8_0);
        case SI_T_Q8_K: return (size_t)(n / SI_QK_K) * sizeof(si_blk_q8_K);
        default: return 0;
    }
}

static void cpu_gemv(si_backend *be, const si_tensor *w, const float *x,
                     float *y, int64_t n_out, si_tp *tp) {
    (void)be;
    const int64_t n_in = (int64_t)w->ne[0];
    uint32_t pair;
    if (!si_quant_vec_dot_pair(w->type, &pair)) {
        si_set_err("cpu gemv: unsupported weight type %s", si_quant_name(w->type));
        return;
    }
    si_vec_dot_fn dot = si_vec_dot_for(w->type);
    if (!dot) {
        si_set_err("cpu gemv: no dot kernel for %s", si_quant_name(w->type));
        return;
    }

    const size_t xq_bytes = pair_row_bytes(pair, n_in);
    void *xq;
    if (pair == SI_T_F32) {
        xq = (void *)(uintptr_t)x;  /* F32 weights consume F32 directly */
    } else {
        xq = si_xmalloc(xq_bytes ? xq_bytes : 1);
        conv_task c = { .x = x, .out = xq, .n = n_in, .pair = pair };
        si_tp_run(tp, conv_task_fn, &c);
    }

    dot_task d = {
        .w = w, .xq = xq, .y = y, .n_out = n_out,
        .row_bytes = (int64_t)si_tensor_row_bytes(w),
        .n_in = n_in, .dot = dot,
    };
    si_tp_run(tp, dot_task_fn, &d);

    if (pair != SI_T_F32) {
        free(xq);
    }
}

static void cpu_row_to_f32(si_backend *be, const si_tensor *w,
                           uint64_t row, float *out) {
    (void)be;
    const int64_t n_in = (int64_t)w->ne[0];
    const char *base = (const char *)w->data +
                       (size_t)row * (size_t)si_tensor_row_bytes(w);
    if (!si_dequant_row(w->type, base, out, (int)n_in)) {
        si_set_err("cpu row_to_f32: unsupported type %s", si_quant_name(w->type));
    }
}

static si_backend g_cpu_backend = {
    .name = "cpu",
    .loc = SI_LOC_HOST,
    .gemv = cpu_gemv,
    .row_to_f32 = cpu_row_to_f32,
};

si_backend *si_backend_cpu(void) {
    return &g_cpu_backend;
}

int si_tensor_from_gguf(const si_gguf *g, const void *gguf_tensor,
                        si_backend *be, si_tensor *out) {
    (void)g;
    const si_gguf_tensor *t = gguf_tensor;
    if (!t || !t->data) {
        si_set_err("tensor not found in file");
        return SI_ENOENT;
    }
    uint32_t blck, tsz;
    if (!si_quant_geometry(t->ggml_type, &blck, &tsz)) {
        si_set_err("unsupported tensor type %s for %s",
                   si_quant_name(t->ggml_type), t->name);
        return SI_ENOTSUP;
    }
    /* blocks span the innermost dimension only (row length); outer dims
     * count blocks and may be any number */
    if (t->n_dims > 0 && t->dims[0] % blck != 0) {
        si_set_err("dim 0 of %s not a multiple of the %s block size",
                   t->name, si_quant_name(t->ggml_type));
        return SI_EFORMAT;
    }
    memset(out, 0, sizeof *out);
    out->type = t->ggml_type;
    for (uint32_t i = 0; i < t->n_dims && i < 4; i++) {
        out->ne[i] = t->dims[i];
    }
    out->n_elems = t->n_elems;
    out->n_bytes = t->n_bytes;
    out->data = t->data;
    out->loc = SI_LOC_HOST;
    out->be = be;
    return SI_OK;
}
