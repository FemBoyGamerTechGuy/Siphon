/* siphon: tensors and backends
 *
 * A tensor is (type, dims, bytes, location). Weights loaded from GGUF
 * stay zero-copy in the mmap and are tagged SI_LOC_HOST; activations
 * live in engine-owned host buffers. The Vulkan milestone adds a
 * device backend and SI_LOC_DEVICE, with per-layer assignment deciding
 * where each layer's weights execute — the engine already routes every
 * op through the backend vtable so no call site changes later.
 *
 * The CPU backend's gemv reproduces the reference's mixed-precision
 * matvec exactly: the activation row is converted once into the weight
 * type's dot-pair format (F16 row / Q8_0 blocks / Q8_K super-blocks,
 * bit-exact quantizers), then each output element is one full dot
 * product computed start-to-finish inside a single lane, which is what
 * makes results independent of the thread count.
 */
#ifndef SIPHON_TENSOR_H
#define SIPHON_TENSOR_H

#include "siphon/common.h"
#include "siphon/gguf.h"
#include "siphon/quant.h"
#include "siphon/threadpool.h"

typedef enum {
    SI_LOC_NONE = 0,
    SI_LOC_HOST = 1,   /* mmap'd weights or host-heap activations */
    /* SI_LOC_DEVICE arrives with the Vulkan backend (milestone 3) */
} si_loc;

typedef struct si_backend si_backend;

typedef struct si_tensor {
    uint32_t type;          /* ggml type id (SI_T_*) */
    uint64_t ne[4];         /* ne[0] = row length */
    uint64_t n_elems;       /* product of ne */
    uint64_t n_bytes;       /* encoded size */
    const void *data;       /* SI_LOC_HOST: mmap pointer */
    si_loc loc;
    si_backend *be;         /* backend this tensor is assigned to */
} si_tensor;

struct si_backend {
    const char *name;
    si_loc loc;

    /* y[i] = <w.row(i), x> for i in [0, n_out). w is [n_out, n_in].
     * The activation conversion happens exactly once, inside. */
    void (*gemv)(si_backend *be, const si_tensor *w, const float *x,
                 float *y, int64_t n_out, si_tp *tp);

    /* Dequantize row `row` of w into out (row length floats). */
    void (*row_to_f32)(si_backend *be, const si_tensor *w,
                       uint64_t row, float *out);
};

/* The host/CPU backend (single instance, no state). */
si_backend *si_backend_cpu(void);

/* Build a tensor view over GGUF tensor info. The data pointer must stay
 * valid (the si_gguf must outlive the model). */
int si_tensor_from_gguf(const si_gguf *g, const void *gguf_tensor,
                        si_backend *be, si_tensor *out);

/* Bytes occupied by one row of a tensor of this type. */
static inline uint64_t si_tensor_row_bytes(const si_tensor *t) {
    uint32_t blck, tsz;
    if (!si_quant_geometry(t->type, &blck, &tsz)) return 0;
    return t->ne[0] / blck * tsz;
}

#endif /* SIPHON_TENSOR_H */
