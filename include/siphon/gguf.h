/* siphon: GGUF container reader
 *
 * Zero-copy, mmap-based parser for GGUF v2/v3 (little-endian) files.
 * Layout reference: ggml GGUF spec (docs/gguf.md), cross-checked
 * against ggml's own reader at development time.
 *
 *   header:      "GGUF" | u32 version | u64 n_tensors | u64 n_kv
 *   kv entries:  str key, u32 value type, value
 *   tensor info: str name, u32 n_dims, u64 dims[n], u32 ggml type, u64 offset
 *   padding to `general.alignment` (default 32), then tensor data.
 *
 * All strings and tensor data point directly into the mmap; only keys
 * and tensor names are copied (they need to be nul-terminated for use
 * as C strings). The mapping stays alive until si_gguf_close().
 */
#ifndef SIPHON_GGUF_H
#define SIPHON_GGUF_H

#include "siphon/common.h"

/* GGUF metadata value types (numbering matches the spec exactly). */
typedef enum {
    SI_GGUF_V_U8     = 0,
    SI_GGUF_V_I8     = 1,
    SI_GGUF_V_U16    = 2,
    SI_GGUF_V_I16    = 3,
    SI_GGUF_V_U32    = 4,
    SI_GGUF_V_I32    = 5,
    SI_GGUF_V_F32    = 6,
    SI_GGUF_V_BOOL   = 7,
    SI_GGUF_V_STRING = 8,
    SI_GGUF_V_ARRAY  = 9,
    SI_GGUF_V_U64    = 10,
    SI_GGUF_V_I64    = 11,
    SI_GGUF_V_F64    = 12,
} si_gguf_vtype;

/* A string that lives inside the mmap (not nul-terminated). */
typedef struct {
    const char *ptr;
    uint64_t len;
} si_gguf_str;

/* One metadata key-value pair. Scalar values are decoded inline; strings
 * and array payloads point into the mmap. */
typedef struct {
    char *key;                 /* owned, nul-terminated */
    si_gguf_vtype type;
    union {
        uint64_t u;            /* U8/U16/U32/U64, and BOOL (0/1) */
        int64_t i;             /* I8/I16/I32/I64 */
        double d;              /* F32/F64 */
        si_gguf_str str;       /* STRING */
        struct {               /* ARRAY */
            si_gguf_vtype elem;
            uint64_t count;
            const uint8_t *data;   /* first element, packed */
        } arr;
    } v;
    /* Lazily-built index for STRING arrays (see si_gguf_arr_str_get). */
    si_gguf_str *str_index;
} si_gguf_kv;

/* GGML tensor types we know the block geometry of. "Known" means we can
 * compute byte sizes; dequantization support is a separate question
 * tracked per milestone. */
typedef struct {
    const char *name;     /* e.g. "Q4_K" */
    uint32_t blck_size;   /* elements per block */
    uint32_t type_size;   /* bytes per block */
    bool known;           /* entry is populated */
} si_gguf_typeinfo;

/* One tensor's info + derived numbers. */
typedef struct {
    char *name;           /* owned, nul-terminated */
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    uint64_t offset;      /* relative to data section start */
    uint64_t n_elems;     /* product of dims */
    uint64_t n_bytes;     /* encoded size in file */
    const uint8_t *data;  /* pointer into mmap (data + offset) */
} si_gguf_tensor;

typedef struct {
    void *map;            /* mmap base, NULL if not mapped */
    size_t map_size;
    uint32_t version;     /* 2 or 3 */
    uint64_t alignment;   /* general.alignment, default 32 */
    uint64_t n_kv;
    si_gguf_kv *kvs;      /* in file order */
    uint64_t n_tensors;
    si_gguf_tensor *tensors;
    const uint8_t *data;      /* start of tensor data section */
    uint64_t data_size;
} si_gguf;

/* Parse a GGUF file. Returns SI_OK or negative error; *out set on success. */
int si_gguf_open(const char *path, si_gguf **out);
void si_gguf_close(si_gguf *g);

/* Key lookup (linear scan, file order). NULL if absent. */
const si_gguf_kv *si_gguf_find(const si_gguf *g, const char *key);

/* Typed getters with defaults. Integer getters accept any integer value
 * type and widen; float getters accept F32/F64 and integer types. */
uint64_t si_gguf_get_u64(const si_gguf *g, const char *key, uint64_t dflt);
int64_t si_gguf_get_i64(const si_gguf *g, const char *key, int64_t dflt);
double si_gguf_get_f64(const si_gguf *g, const char *key, double dflt);
bool si_gguf_get_bool(const si_gguf *g, const char *key, bool dflt);
/* Returns NULL if absent; pointer valid while the si_gguf is open. */
si_gguf_str si_gguf_get_str(const si_gguf *g, const char *key);

/* Array element access.
 * si_gguf_arr_str_get is O(1) after the first call (builds a cached index).
 * Numeric arrays: data pointer + count, element stride = size of type. */
int si_gguf_arr_str_get(si_gguf *g, const char *key, uint64_t idx,
                        si_gguf_str *out);
const si_gguf_kv *si_gguf_find_arr(const si_gguf *g, const char *key);

/* Find tensor by name (linear scan). NULL if absent. */
const si_gguf_tensor *si_gguf_tensor_find(const si_gguf *g, const char *name);

/* ggml type geometry. Unknown enum values return known=false and
 * name "TYPE_x". */
si_gguf_typeinfo si_gguf_type_info(uint32_t ggml_type);
const char *si_gguf_vtype_name(si_gguf_vtype t);

/* Value of a kv for printing: renders scalars/short strings into buf. */
void si_gguf_kv_render(const si_gguf_kv *kv, char *buf, size_t buflen);

#endif /* SIPHON_GGUF_H */
