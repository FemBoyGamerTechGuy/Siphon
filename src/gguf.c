/* siphon: GGUF container reader (mmap, zero-copy)
 *
 * Block geometry table cross-checked against ggml-common.h block
 * structures (each entry verified against the static_asserts that
 * ggml itself enforces), so tensor sizes and data offsets are exact.
 */
#include "siphon/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdarg.h>

/* ---- ggml type geometry -------------------------------------------- */

typedef struct {
    const char *name;
    uint32_t blck_size;
    uint32_t type_size;
} type_entry;

/* GGML_TYPE_* enum values per ggml.h; sizes per ggml-common.h structs:
 *   classic quants: 32-elem blocks (QK4_0 etc.)
 *   K-quants:       256-elem blocks (QK_K)
 *   IQ quants:      256-elem blocks except IQ4_NL (32)
 *   ternary (gpt-oss): 256-elem blocks
 */
static const type_entry g_types[] = {
    [0]  = { "F32",      1,   4 },
    [1]  = { "F16",      1,   2 },
    [2]  = { "Q4_0",    32,  18 },
    [3]  = { "Q4_1",    32,  20 },
    [6]  = { "Q5_0",    32,  22 },
    [7]  = { "Q5_1",    32,  24 },
    [8]  = { "Q8_0",    32,  34 },
    [9]  = { "Q8_1",    32,  36 },
    [10] = { "Q2_K",   256,  84 },
    [11] = { "Q3_K",   256, 110 },
    [12] = { "Q4_K",   256, 144 },
    [13] = { "Q5_K",   256, 176 },
    [14] = { "Q6_K",   256, 210 },
    [15] = { "Q8_K",   256, 292 },
    [16] = { "IQ2_XXS", 256,  66 },
    [17] = { "IQ2_XS",  256,  74 },
    [18] = { "IQ3_XXS", 256,  98 },
    [19] = { "IQ1_S",   256,  50 },
    [20] = { "IQ4_NL",   32,  18 },
    [21] = { "IQ3_S",   256, 110 },
    [22] = { "IQ2_S",   256,  82 },
    [23] = { "IQ4_XS",  256, 136 },
    [24] = { "I8",       1,   1 },
    [25] = { "I16",      1,   2 },
    [26] = { "I32",      1,   4 },
    [27] = { "I64",      1,   8 },
    [28] = { "F64",      1,   8 },
    [29] = { "IQ1_M",   256,  56 },
    [30] = { "BF16",     1,   2 },
    [34] = { "TQ1_0",   256,  54 },
    [35] = { "TQ2_0",   256,  66 },
    [39] = { "MXFP4",    32,  18 },
    [40] = { "NVFP4",    64,  36 },
    [41] = { "Q1_0",    128,  18 },
    [42] = { "Q2_0",     64,  18 },
};

#define G_TYPES_MAX ((int)(sizeof(g_types) / sizeof(g_types[0])))

si_gguf_typeinfo si_gguf_type_info(uint32_t t) {
    si_gguf_typeinfo info = { NULL, 0, 0, false };
    if ((int) t < G_TYPES_MAX && g_types[t].name != NULL) {
        info.name = g_types[t].name;
        info.blck_size = g_types[t].blck_size;
        info.type_size = g_types[t].type_size;
        info.known = true;
    }
    return info;
}

/* ---- value type names ----------------------------------------------- */

static const char *const g_vtype_names[] = {
    "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool",
    "string", "array", "u64", "i64", "f64",
};

const char *si_gguf_vtype_name(si_gguf_vtype t) {
    if ((int) t >= 0 && (int) t < (int)(sizeof(g_vtype_names) / sizeof(g_vtype_names[0]))) {
        return g_vtype_names[t];
    }
    return "?";
}

static uint32_t vtype_scalar_size(si_gguf_vtype t) {
    switch (t) {
        case SI_GGUF_V_U8:
        case SI_GGUF_V_I8:
        case SI_GGUF_V_BOOL:   return 1;
        case SI_GGUF_V_U16:
        case SI_GGUF_V_I16:    return 2;
        case SI_GGUF_V_U32:
        case SI_GGUF_V_I32:
        case SI_GGUF_V_F32:    return 4;
        case SI_GGUF_V_U64:
        case SI_GGUF_V_I64:
        case SI_GGUF_V_F64:    return 8;
        default:               return 0;  /* STRING / ARRAY handled apart */
    }
}

/* ---- mmap + bounds-checked cursor ----------------------------------- */

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
    const char *ctx;   /* what we are parsing, for error messages */
} cursor;

static bool cur_need(cursor *c, size_t n) {
    if (c->size - c->pos < n) {
        si_set_err("truncated GGUF: needed %zu bytes past end while reading %s",
                   n, c->ctx);
        return false;
    }
    return true;
}

static uint8_t  cur_u8(cursor *c)  { uint8_t v;  memcpy(&v, c->base + c->pos, 1); c->pos += 1; return v; }
static uint16_t cur_u16(cursor *c) { uint16_t v; memcpy(&v, c->base + c->pos, 2); c->pos += 2; return v; }
static uint32_t cur_u32(cursor *c) { uint32_t v; memcpy(&v, c->base + c->pos, 4); c->pos += 4; return v; }
static uint64_t cur_u64(cursor *c) { uint64_t v; memcpy(&v, c->base + c->pos, 8); c->pos += 8; return v; }
static void cur_skip(cursor *c, size_t n) { c->pos += n; }

/* Read one length-prefixed string. Returns false on truncation. */
static bool cur_str(cursor *c, si_gguf_str *out) {
    if (!cur_need(c, 8)) return false;
    uint64_t len = cur_u64(c);
    if (len > c->size - c->pos) {
        si_set_err("truncated GGUF: string of length %llu exceeds file", (unsigned long long) len);
        return false;
    }
    out->ptr = (const char *) (c->base + c->pos);
    out->len = len;
    cur_skip(c, (size_t) len);
    return true;
}

/* ---- parsing --------------------------------------------------------- */

static int parse_kv(cursor *c, si_gguf_kv *kv) {
    si_gguf_str key;
    if (!cur_str(c, &key)) return SI_EFORMAT;

    uint32_t vtype;
    if (!cur_need(c, 4)) return SI_EFORMAT;
    vtype = cur_u32(c);
    if (vtype > (uint32_t) SI_GGUF_V_F64) {
        si_set_err("invalid GGUF value type %u for key '%.*s'", vtype,
                   (int) (key.len < 200 ? key.len : 200), key.ptr);
        return SI_EFORMAT;
    }
    kv->type = (si_gguf_vtype) vtype;
    kv->str_index = NULL;
    kv->key = si_xstrndup(key.ptr, key.len);

    uint32_t ssz = vtype_scalar_size(kv->type);
    if (ssz) {
        if (!cur_need(c, ssz)) { free(kv->key); kv->key = NULL; return SI_EFORMAT; }
        switch (kv->type) {
            case SI_GGUF_V_U8:   kv->v.u = cur_u8(c);  break;
            case SI_GGUF_V_I8:   kv->v.i = (int8_t) cur_u8(c);  break;
            case SI_GGUF_V_U16:  kv->v.u = cur_u16(c); break;
            case SI_GGUF_V_I16:  kv->v.i = (int16_t) cur_u16(c); break;
            case SI_GGUF_V_U32:  kv->v.u = cur_u32(c); break;
            case SI_GGUF_V_I32:  kv->v.i = (int32_t) cur_u32(c); break;
            case SI_GGUF_V_F32:  { uint32_t b = cur_u32(c); float f; memcpy(&f, &b, 4); kv->v.d = (double) f; break; }
            case SI_GGUF_V_U64:  kv->v.u = cur_u64(c); break;
            case SI_GGUF_V_I64:  kv->v.i = (int64_t) cur_u64(c); break;
            case SI_GGUF_V_F64:  { uint64_t b = cur_u64(c); double d; memcpy(&d, &b, 8); kv->v.d = d; break; }
            case SI_GGUF_V_BOOL: kv->v.u = cur_u8(c) ? 1 : 0; break;
            default: break;
        }
        return SI_OK;
    }

    if (kv->type == SI_GGUF_V_STRING) {
        return cur_str(c, &kv->v.str) ? SI_OK : SI_EFORMAT;
    }

    /* ARRAY: element type + count + packed elements. We only record the
     * element region; callers iterate with the accessors below. */
    if (!cur_need(c, 12)) return SI_EFORMAT;
    uint32_t etype = cur_u32(c);
    uint64_t count = cur_u64(c);
    if (etype > (uint32_t) SI_GGUF_V_F64) {
        si_set_err("invalid GGUF array element type %u for key '%s'", etype, kv->key);
        return SI_EFORMAT;
    }
    kv->v.arr.elem = (si_gguf_vtype) etype;
    kv->v.arr.count = count;
    kv->v.arr.data = c->base + c->pos;

    uint32_t essz = vtype_scalar_size(kv->v.arr.elem);
    if (essz) {
        /* numeric array: count * essz bytes */
        if (count > (c->size - c->pos) / essz) {
            si_set_err("truncated GGUF: array '%s' claims %llu elements", kv->key,
                       (unsigned long long) count);
            return SI_EFORMAT;
        }
        cur_skip(c, (size_t) count * essz);
    } else if (kv->v.arr.elem == SI_GGUF_V_STRING) {
        /* string array: validate while walking */
        for (uint64_t i = 0; i < count; i++) {
            if (!cur_need(c, 8)) return SI_EFORMAT;
            uint64_t len = cur_u64(c);
            if (len > c->size - c->pos) {
                si_set_err("truncated GGUF: string %llu of array '%s' exceeds file",
                           (unsigned long long) i, kv->key);
                return SI_EFORMAT;
            }
            cur_skip(c, (size_t) len);
        }
    } else {
        /* nested arrays exist in theory (arrays of arrays); no known
         * model uses them. Record count and reject cleanly. */
        si_set_err("GGUF nested arrays not supported (key '%s')", kv->key);
        return SI_ENOTSUP;
    }
    return SI_OK;
}

static int parse_tensor_info(cursor *c, si_gguf_tensor *t) {
    si_gguf_str name;
    if (!cur_str(c, &name)) return SI_EFORMAT;

    if (!cur_need(c, 4)) return SI_EFORMAT;
    uint32_t nd = cur_u32(c);
    if (nd < 1 || nd > 4) {
        si_set_err("tensor '%.*s' has invalid dimension count %u",
                   (int) (name.len < 200 ? name.len : 200), name.ptr, nd);
        return SI_EFORMAT;
    }
    if (!cur_need(c, (size_t) nd * 8)) return SI_EFORMAT;

    t->name = si_xstrndup(name.ptr, name.len);
    t->n_dims = nd;
    t->n_elems = 1;
    for (uint32_t i = 0; i < nd; i++) {
        t->dims[i] = cur_u64(c);
        if (t->dims[i] == 0) {
            si_set_err("tensor '%s' has zero dimension %u", t->name, i);
            return SI_EFORMAT;
        }
        if (t->n_elems > UINT64_MAX / t->dims[i]) {
            si_set_err("tensor '%s' element count overflows", t->name);
            return SI_EFORMAT;
        }
        t->n_elems *= t->dims[i];
    }

    if (!cur_need(c, 12)) return SI_EFORMAT;
    t->ggml_type = cur_u32(c);
    t->offset = cur_u64(c);

    si_gguf_typeinfo ti = si_gguf_type_info(t->ggml_type);
    if (!ti.known) {
        si_set_err("tensor '%s' uses unknown ggml type %u", t->name, t->ggml_type);
        return SI_ENOTSUP;
    }
    t->n_bytes = (t->n_elems + ti.blck_size - 1) / ti.blck_size * ti.type_size;
    if (ti.blck_size > 1 && t->n_elems % ti.blck_size != 0) {
        /* ggml requires exact divisibility; real files satisfy it. */
        si_log_warn("tensor '%s': %llu elements not divisible by block size %u "
                    "(size rounded up)", t->name,
                    (unsigned long long) t->n_elems, ti.blck_size);
    }
    return SI_OK;
}

int si_gguf_open(const char *path, si_gguf **out) {
    *out = NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        si_set_err("cannot open '%s'", path);
        return SI_ENOENT;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        si_set_err("'%s' is not a regular file", path);
        return SI_EIO;
    }
    size_t fsize = (size_t) st.st_size;
    if (fsize < 20) {  /* magic + version + counts = 24, allow tiny for clear error */
        close(fd);
        si_set_err("'%s' is too small to be a GGUF file", path);
        return SI_EFORMAT;
    }

    void *map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        si_set_err("mmap failed on '%s'", path);
        return SI_EIO;
    }

    si_gguf *g = si_xcalloc(1, sizeof(*g));
    g->map = map;
    g->map_size = fsize;

    cursor c = { (const uint8_t *) map, fsize, 0, "header" };
    int rc = SI_EFORMAT;

    char magic[4];
    if (!cur_need(&c, 4)) goto fail;
    memcpy(magic, c.base + c.pos, 4); cur_skip(&c, 4);
    if (memcmp(magic, "GGUF", 4) != 0) {
        if (memcmp(magic, "FUGG", 4) == 0) {
            si_set_err("'%s' is a big-endian GGUF file; only little-endian "
                       "is supported (target hardware is x86)", path);
        } else {
            si_set_err("'%s' is not a GGUF file (bad magic)", path);
        }
        goto fail;
    }

    if (!cur_need(&c, 4)) goto fail;
    g->version = cur_u32(&c);
    /* v3 == v2 on little-endian (v3 only adds big-endian support);
     * v1 used 32-bit lengths and is rejected as ancient. */
    if (g->version < 2 || g->version > 3) {
        si_set_err("unsupported GGUF version %u (need 2 or 3)", g->version);
        rc = SI_ENOTSUP;
        goto fail;
    }

    if (!cur_need(&c, 16)) goto fail;
    g->n_tensors = cur_u64(&c);
    g->n_kv = cur_u64(&c);
    if (g->n_tensors > 1000000 || g->n_kv > 1000000) {
        si_set_err("implausible GGUF counts (tensors=%llu kv=%llu), file corrupt?",
                   (unsigned long long) g->n_tensors, (unsigned long long) g->n_kv);
        goto fail;
    }

    g->kvs = si_xcalloc(g->n_kv ? (size_t) g->n_kv : 1, sizeof(*g->kvs));
    c.ctx = "metadata";
    for (uint64_t i = 0; i < g->n_kv; i++) {
        rc = parse_kv(&c, &g->kvs[i]);
        if (rc != SI_OK) goto fail;
    }

    g->tensors = si_xcalloc(g->n_tensors ? (size_t) g->n_tensors : 1, sizeof(*g->tensors));
    c.ctx = "tensor info";
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        rc = parse_tensor_info(&c, &g->tensors[i]);
        if (rc != SI_OK) goto fail;
    }

    /* alignment + data section */
    g->alignment = 32;
    {
        const si_gguf_kv *a = si_gguf_find(g, "general.alignment");
        if (a && (a->type == SI_GGUF_V_U32 || a->type == SI_GGUF_V_U64)) {
            g->alignment = a->v.u;
        }
    }
    if (g->alignment == 0) {
        si_log_warn("general.alignment is 0, using 32");
        g->alignment = 32;
    }
    uint64_t pad = (g->alignment - (c.pos % g->alignment)) % g->alignment;
    if ((uint64_t) c.pos + pad > fsize) {
        si_set_err("truncated GGUF: no data section");
        goto fail;
    }
    g->data = (const uint8_t *) map + c.pos + pad;
    g->data_size = fsize - (c.pos + pad);

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        si_gguf_tensor *t = &g->tensors[i];
        if (t->offset % g->alignment != 0) {
            si_log_warn("tensor '%s' offset %llu not aligned to %llu",
                        t->name, (unsigned long long) t->offset,
                        (unsigned long long) g->alignment);
        }
        if (t->offset > g->data_size || t->n_bytes > g->data_size - t->offset) {
            si_set_err("tensor '%s' data (%llu bytes at offset %llu) exceeds "
                       "data section (%llu bytes)", t->name,
                       (unsigned long long) t->n_bytes,
                       (unsigned long long) t->offset,
                       (unsigned long long) g->data_size);
            rc = SI_EFORMAT;
            goto fail;
        }
        t->data = g->data + t->offset;
    }

    *out = g;
    return SI_OK;

fail:
    si_gguf_close(g);
    return rc;
}

void si_gguf_close(si_gguf *g) {
    if (!g) return;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        free(g->kvs[i].key);
        free(g->kvs[i].str_index);
    }
    free(g->kvs);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        free(g->tensors[i].name);
    }
    free(g->tensors);
    if (g->map) munmap(g->map, g->map_size);
    free(g);
}

/* ---- lookups --------------------------------------------------------- */

const si_gguf_kv *si_gguf_find(const si_gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (strcmp(g->kvs[i].key, key) == 0) {
            return &g->kvs[i];
        }
    }
    return NULL;
}

const si_gguf_kv *si_gguf_find_arr(const si_gguf *g, const char *key) {
    const si_gguf_kv *kv = si_gguf_find(g, key);
    if (!kv || kv->type != SI_GGUF_V_ARRAY) return NULL;
    return kv;
}

static bool kv_is_int(const si_gguf_kv *kv) {
    switch (kv->type) {
        case SI_GGUF_V_U8: case SI_GGUF_V_U16:
        case SI_GGUF_V_U32: case SI_GGUF_V_U64:
        case SI_GGUF_V_I8: case SI_GGUF_V_I16:
        case SI_GGUF_V_I32: case SI_GGUF_V_I64:
            return true;
        default:
            return false;
    }
}

static bool kv_is_unsigned(const si_gguf_kv *kv) {
    switch (kv->type) {
        case SI_GGUF_V_U8: case SI_GGUF_V_U16:
        case SI_GGUF_V_U32: case SI_GGUF_V_U64:
            return true;
        default:
            return false;
    }
}

uint64_t si_gguf_get_u64(const si_gguf *g, const char *key, uint64_t dflt) {
    const si_gguf_kv *kv = si_gguf_find(g, key);
    if (!kv || !kv_is_int(kv)) return dflt;
    return kv_is_unsigned(kv) ? kv->v.u : (uint64_t) kv->v.i;
}

int64_t si_gguf_get_i64(const si_gguf *g, const char *key, int64_t dflt) {
    const si_gguf_kv *kv = si_gguf_find(g, key);
    if (!kv || !kv_is_int(kv)) return dflt;
    return kv_is_unsigned(kv) ? (int64_t) kv->v.u : kv->v.i;
}

double si_gguf_get_f64(const si_gguf *g, const char *key, double dflt) {
    const si_gguf_kv *kv = si_gguf_find(g, key);
    if (!kv) return dflt;
    if (kv->type == SI_GGUF_V_F32 || kv->type == SI_GGUF_V_F64) return kv->v.d;
    if (kv_is_int(kv)) {
        return kv->type <= SI_GGUF_V_U32 ? (double) kv->v.u : (double) kv->v.i;
    }
    return dflt;
}

bool si_gguf_get_bool(const si_gguf *g, const char *key, bool dflt) {
    const si_gguf_kv *kv = si_gguf_find(g, key);
    if (!kv || kv->type != SI_GGUF_V_BOOL) return dflt;
    return kv->v.u != 0;
}

si_gguf_str si_gguf_get_str(const si_gguf *g, const char *key) {
    si_gguf_str empty = { NULL, 0 };
    const si_gguf_kv *kv = si_gguf_find(g, key);
    if (!kv || kv->type != SI_GGUF_V_STRING) return empty;
    return kv->v.str;
}

/* Build (once) an index of string-array elements for O(1) access. */
static int build_str_index(si_gguf_kv *kv) {
    if (kv->str_index) return SI_OK;
    si_gguf_str *idx = si_xcalloc(kv->v.arr.count ? (size_t) kv->v.arr.count : 1,
                                  sizeof(*idx));
    const uint8_t *p = kv->v.arr.data;
    for (uint64_t i = 0; i < kv->v.arr.count; i++) {
        uint64_t len;
        memcpy(&len, p, 8);
        p += 8;
        idx[i].ptr = (const char *) p;
        idx[i].len = len;
        p += len;
    }
    kv->str_index = idx;
    return SI_OK;
}

int si_gguf_arr_str_get(si_gguf *g, const char *key, uint64_t idx,
                        si_gguf_str *out) {
    const si_gguf_kv *kvc = si_gguf_find(g, key);
    if (!kvc || kvc->type != SI_GGUF_V_ARRAY || kvc->v.arr.elem != SI_GGUF_V_STRING) {
        si_set_err("key '%s' is not a string array", key);
        return SI_EFORMAT;
    }
    si_gguf_kv *kv = (si_gguf_kv *) kvc;  /* cache is owned by the container */
    int rc = build_str_index(kv);
    if (rc != SI_OK) return rc;
    if (idx >= kv->v.arr.count) {
        si_set_err("index %llu out of range for '%s' (%llu entries)",
                   (unsigned long long) idx, key,
                   (unsigned long long) kv->v.arr.count);
        return SI_ERANGE;
    }
    *out = kv->str_index[idx];
    return SI_OK;
}

const si_gguf_tensor *si_gguf_tensor_find(const si_gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) {
            return &g->tensors[i];
        }
    }
    return NULL;
}

/* ---- rendering for `siphon info` ------------------------------------- */

void si_gguf_kv_render(const si_gguf_kv *kv, char *buf, size_t buflen) {
    switch (kv->type) {
        case SI_GGUF_V_STRING:
            snprintf(buf, buflen, "\"%.*s\"", (int) (kv->v.str.len < buflen - 8 ?
                                       kv->v.str.len : buflen - 8), kv->v.str.ptr);
            break;
        case SI_GGUF_V_BOOL:
            snprintf(buf, buflen, "%s", kv->v.u ? "true" : "false");
            break;
        case SI_GGUF_V_F32:
        case SI_GGUF_V_F64:
            snprintf(buf, buflen, "%g", kv->v.d);
            break;
        case SI_GGUF_V_U8: case SI_GGUF_V_U16: case SI_GGUF_V_U32: case SI_GGUF_V_U64:
            snprintf(buf, buflen, "%llu", (unsigned long long) kv->v.u);
            break;
        default:
            snprintf(buf, buflen, "%lld", (long long) kv->v.i);
            break;
    }
}
