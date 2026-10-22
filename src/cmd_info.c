/* siphon: `siphon info <model>` — inspect a GGUF file.
 *
 * Prints the container summary, general + architecture metadata,
 * tokenizer summary, and tensor statistics. Reading is mmap-lazy, so
 * this is near-instant even on multi-hundred-GB files.
 */
#include "siphon/gguf.h"

#include <stdarg.h>

typedef struct {
    uint32_t ggml_type;
    uint64_t count;
    uint64_t bytes;
} type_stat;

static const char *file_type_name(uint32_t ft) {
    /* llama.cpp LLM_FTYPE values (stable ABI). */
    switch (ft) {
        case 0:  return "ALL_F32";
        case 1:  return "MOSTLY_F16";
        case 2:  return "MOSTLY_Q4_0";
        case 3:  return "MOSTLY_Q4_1";
        case 7:  return "MOSTLY_Q8_0";
        case 8:  return "MOSTLY_Q5_0";
        case 9:  return "MOSTLY_Q5_1";
        case 10: return "MOSTLY_Q2_K";
        case 11: return "MOSTLY_Q3_K_S";
        case 12: return "MOSTLY_Q3_K_M";
        case 13: return "MOSTLY_Q3_K_L";
        case 14: return "MOSTLY_Q4_K_S";
        case 15: return "MOSTLY_Q4_K_M";
        case 16: return "MOSTLY_Q5_K_S";
        case 17: return "MOSTLY_Q5_K_M";
        case 18: return "MOSTLY_Q6_K";
        case 19: return "MOSTLY_IQ2_XXS";
        case 20: return "MOSTLY_IQ2_XS";
        case 21: return "MOSTLY_Q2_K_S";
        case 22: return "MOSTLY_IQ3_XS";
        case 23: return "MOSTLY_IQ3_XXS";
        case 24: return "MOSTLY_IQ1_S";
        case 25: return "MOSTLY_IQ4_NL";
        case 26: return "MOSTLY_IQ3_S";
        case 27: return "MOSTLY_IQ3_M";
        case 28: return "MOSTLY_IQ2_S";
        case 29: return "MOSTLY_IQ2_M";
        case 30: return "MOSTLY_IQ4_XS";
        case 31: return "MOSTLY_IQ1_M";
        case 32: return "MOSTLY_BF16";
        case 36: return "MOSTLY_TQ1_0";
        case 37: return "MOSTLY_TQ2_0";
        case 38: return "MOSTLY_MXFP4_MOE";
        case 39: return "MOSTLY_NVFP4";
        default: return NULL;
    }
}

static void print_kv_group(const si_gguf *g, const char *prefix) {
    size_t plen = strlen(prefix);
    bool header_done = false;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        const si_gguf_kv *kv = &g->kvs[i];
        if (strncmp(kv->key, prefix, plen) != 0) continue;
        const char *rest = kv->key + plen;
        if (rest[0] == '\0') continue;  /* the prefix key itself */
        if (!header_done) {
            printf("\n%s\n", prefix);
            header_done = true;
        }
        char val[96];
        si_gguf_kv_render(kv, val, sizeof(val));
        if (kv->type == SI_GGUF_V_ARRAY) {
            snprintf(val, sizeof(val), "array[%llu] of %s",
                     (unsigned long long) kv->v.arr.count,
                     si_gguf_vtype_name(kv->v.arr.elem));
        }
        printf("  %-45s = %s\n", rest, val);
    }
}

static void print_tokenizer_section(const si_gguf *g) {
    si_gguf_str model = si_gguf_get_str(g, "tokenizer.ggml.model");
    if (!model.ptr) return;

    printf("\ntokenizer\n");
    printf("  %-45s = %.*s\n", "model", (int) model.len, model.ptr);

    si_gguf_str pre = si_gguf_get_str(g, "tokenizer.ggml.pre");
    if (pre.ptr) {
        printf("  %-45s = %.*s\n", "pre", (int) pre.len, pre.ptr);
    }

    const si_gguf_kv *tokens = si_gguf_find_arr(g, "tokenizer.ggml.tokens");
    if (tokens) {
        printf("  %-45s = %llu\n", "tokens",
               (unsigned long long) tokens->v.arr.count);
    }
    const si_gguf_kv *merges = si_gguf_find_arr(g, "tokenizer.ggml.merges");
    if (merges) {
        printf("  %-45s = %llu\n", "merges",
               (unsigned long long) merges->v.arr.count);
    }
    const si_gguf_kv *scores = si_gguf_find_arr(g, "tokenizer.ggml.scores");
    if (scores) {
        printf("  %-45s = array[%llu] of %s\n", "scores",
               (unsigned long long) scores->v.arr.count,
               si_gguf_vtype_name(scores->v.arr.elem));
    }

    static const struct { const char *key; const char *label; } ids[] = {
        { "tokenizer.ggml.bos_token_id",     "bos token" },
        { "tokenizer.ggml.eos_token_id",     "eos token" },
        { "tokenizer.ggml.unknown_token_id", "unk token" },
        { "tokenizer.ggml.pad_token_id",     "pad token" },
    };
    for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); i++) {
        const si_gguf_kv *kv = si_gguf_find(g, ids[i].key);
        if (!kv) continue;
        int64_t id = si_gguf_get_i64(g, ids[i].key, -1);
        char shown[64];
        if (id >= 0 && tokens) {
            si_gguf_str t;
            if (si_gguf_arr_str_get((si_gguf *) g, "tokenizer.ggml.tokens",
                                    (uint64_t) id, &t) == SI_OK) {
                snprintf(shown, sizeof(shown), "%lld '%.*s'", (long long) id,
                         (int) (t.len > 24 ? 24 : t.len), t.ptr);
            } else {
                snprintf(shown, sizeof(shown), "%lld", (long long) id);
            }
        } else {
            snprintf(shown, sizeof(shown), "%lld", (long long) id);
        }
        printf("  %-45s = %s\n", ids[i].label, shown);
    }

    printf("  %-45s = %s\n", "add_bos_token",
           si_gguf_get_bool(g, "tokenizer.ggml.add_bos_token", false) ? "true" : "false");
    printf("  %-45s = %s\n", "add_eos_token",
           si_gguf_get_bool(g, "tokenizer.ggml.add_eos_token", false) ? "true" : "false");

    si_gguf_str tmpl = si_gguf_get_str(g, "tokenizer.chat_template");
    printf("  %-45s = %s\n", "chat_template",
           tmpl.ptr ? "present (Jinja subset engine lands in milestone 7)" : "absent");
}

static int cmp_stat_bytes(const void *a, const void *b) {
    const type_stat *x = a, *y = b;
    return x->bytes < y->bytes ? 1 : (x->bytes > y->bytes ? -1 : 0);
}

int cmd_info(int argc, char **argv) {
    bool show_tensors = false;
    const char *path = NULL;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--tensors") == 0) {
            show_tensors = true;
        } else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            si_log_warn("unknown flag '%s' for info", argv[i]);
            return 1;
        } else {
            path = argv[i];
        }
    }
    if (!path) {
        fprintf(stderr, "usage: siphon info [--tensors] <model.gguf>\n");
        return 1;
    }

    double t0 = si_time_sec();
    si_gguf *g = NULL;
    int rc = si_gguf_open(path, &g);
    if (rc != SI_OK) {
        si_log_err("%s", si_errstr());
        return 1;
    }
    double parse_ms = (si_time_sec() - t0) * 1000.0;

    char sizebuf[32];
    printf("file\n");
    printf("  %-45s = %s\n", "path", path);
    printf("  %-45s = %s (%llu bytes)\n", "size",
           si_fmt_size(sizebuf, sizeof(sizebuf), g->map_size),
           (unsigned long long) g->map_size);
    printf("  %-45s = %u, %llu tensors, %llu kv entries\n", "gguf",
           g->version, (unsigned long long) g->n_tensors,
           (unsigned long long) g->n_kv);
    printf("  %-45s = %llu\n", "alignment",
           (unsigned long long) g->alignment);
    printf("  %-45s = %.2f ms\n", "metadata parse time", parse_ms);

    si_gguf_str arch = si_gguf_get_str(g, "general.architecture");
    si_gguf_str name = si_gguf_get_str(g, "general.name");
    print_kv_group(g, "general.");
    {
        int64_t ft = si_gguf_get_i64(g, "general.file_type", -1);
        if (ft >= 0 && ft <= 0xFFFF) {
            const char *ftname = file_type_name((uint32_t) ft);
            printf("  %-45s = %lld (%s)\n", "file_type (named)",
                   (long long) ft, ftname ? ftname : "unknown");
        }
    }
    if (arch.ptr && arch.len > 0) {
        char prefix[96];
        snprintf(prefix, sizeof(prefix), "%.*s.", (int) (arch.len < 90 ? arch.len : 90), arch.ptr);
        print_kv_group(g, prefix);
        /* MoE hint: qwen2moe/mixtral etc. carry expert counts. */
    }
    (void) name;

    print_tokenizer_section(g);

    /* tensor statistics */
    uint64_t total_elems = 0, total_bytes = 0;
    type_stat stats[64] = {0};
    uint32_t n_stats = 0;

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const si_gguf_tensor *t = &g->tensors[i];
        total_elems += t->n_elems;
        total_bytes += t->n_bytes;
        type_stat *slot = NULL;
        for (uint32_t s = 0; s < n_stats; s++) {
            if (stats[s].ggml_type == t->ggml_type) { slot = &stats[s]; break; }
        }
        if (!slot && n_stats < 64) {
            slot = &stats[n_stats++];
            slot->ggml_type = t->ggml_type;
        }
        if (slot) { slot->count++; slot->bytes += t->n_bytes; }
    }
    qsort(stats, n_stats, sizeof(stats[0]), cmp_stat_bytes);

    printf("\ntensors\n");
    printf("  %-45s = %llu\n", "count", (unsigned long long) g->n_tensors);
    printf("  %-45s = %s\n", "data section",
           si_fmt_size(sizebuf, sizeof(sizebuf), g->data_size));
    printf("  %-45s = %s\n", "tensor bytes total",
           si_fmt_size(sizebuf, sizeof(sizebuf), total_bytes));
    if (total_elems) {
        double bpw = (double) total_bytes * 8.0 / (double) total_elems;
        char pbuf[32];
        double params = (double) total_elems;
        if (params >= 1e9)      snprintf(pbuf, sizeof(pbuf), "%.2fB", params / 1e9);
        else if (params >= 1e6) snprintf(pbuf, sizeof(pbuf), "%.2fM", params / 1e6);
        else                    snprintf(pbuf, sizeof(pbuf), "%.2fK", params / 1e3);
        printf("  %-45s = %s (sum of tensor elements)\n", "parameters", pbuf);
        printf("  %-45s = %.3f bits/weight average\n", "avg density", bpw);
    }
    for (uint32_t s = 0; s < n_stats; s++) {
        si_gguf_typeinfo ti = si_gguf_type_info(stats[s].ggml_type);
        char line[128];
        snprintf(line, sizeof(line), "%-8s x %-4llu  %10llu bytes  (%.3f bpw)",
                 ti.name, (unsigned long long) stats[s].count,
                 (unsigned long long) stats[s].bytes,
                 ti.blck_size ?
                     (double) ti.type_size * 8.0 / (double) ti.blck_size : 0.0);
        printf("  %-45s   %s\n", s == 0 ? "by bytes:" : "", line);
    }

    if (show_tensors) {
        printf("\ntensor list\n");
        printf("  %-4s  %-32s %-8s %-24s %s\n", "#", "name", "type", "shape", "bytes");
        for (uint64_t i = 0; i < g->n_tensors; i++) {
            const si_gguf_tensor *t = &g->tensors[i];
            si_gguf_typeinfo ti = si_gguf_type_info(t->ggml_type);
            char shape[64];
            size_t off = 0;
            for (uint32_t d = 0; d < t->n_dims; d++) {
                off += (size_t) snprintf(shape + off, sizeof(shape) - off,
                                          "%s%llu", d ? "," : "[",
                                          (unsigned long long) t->dims[d]);
                if (off >= sizeof(shape) - 1) break;
            }
            if (off < sizeof(shape) - 2) snprintf(shape + off, sizeof(shape) - off, "]");
            printf("  %-4llu  %-32s %-8s %-24s %llu\n",
                   (unsigned long long) i, t->name, ti.name, shape,
                   (unsigned long long) t->n_bytes);
        }
    }

    si_gguf_close(g);
    return 0;
}
