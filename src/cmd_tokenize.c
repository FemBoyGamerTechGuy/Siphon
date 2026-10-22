/* siphon: `siphon tokenize` — tokenizer verification tool.
 *
 * Reads text from stdin (or -t), prints token ids, pieces and a
 * roundtrip check. --detok goes the other way (ids on stdin). This is
 * the tool used to verify token-by-token equality with the reference
 * implementation during development.
 */
#include "siphon/vocab.h"

static void print_escaped(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char) s[i];
        if (c == '\n') printf("\\n");
        else if (c == '\r') printf("\\r");
        else if (c == '\t') printf("\\t");
        else if (c < 0x20 || c == 0x7f) printf("\\x%02x", c);
        else putchar(s[i]);
    }
}

static char *read_all_stdin(size_t *len) {
    size_t cap = 1 << 16, n = 0;
    char *buf = si_xmalloc(cap);
    for (;;) {
        if (n == cap) {
            cap *= 2;
            buf = si_xrealloc(buf, cap);
        }
        size_t k = fread(buf + n, 1, cap - n, stdin);
        n += k;
        if (k == 0) break;
    }
    *len = n;
    return buf;
}

int cmd_tokenize(int argc, char **argv) {
    const char *path = NULL;
    const char *text_arg = NULL;
    bool ids_only = false;
    bool add_special = true;
    bool parse_special = false;
    bool detok = false;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--ids") == 0) ids_only = true;
        else if (strcmp(argv[i], "--no-special") == 0) add_special = false;
        else if (strcmp(argv[i], "--parse-special") == 0) parse_special = true;
        else if (strcmp(argv[i], "--detok") == 0) detok = true;
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) text_arg = argv[++i];
        else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            si_log_warn("unknown flag '%s' for tokenize", argv[i]);
            return 1;
        } else path = argv[i];
    }
    if (!path) {
        fprintf(stderr, "usage: siphon tokenize [options] <model.gguf>\n"
                        "  -t <text>        tokenize this string (else stdin)\n"
                        "  --ids            print ids only\n"
                        "  --no-special     don't add bos/eos\n"
                        "  --parse-special  parse special tokens in text\n"
                        "  --detok          decode ids (one per line / space separated) from stdin\n");
        return 1;
    }

    si_gguf *g = NULL;
    if (si_gguf_open(path, &g) != SI_OK) {
        si_log_err("%s", si_errstr());
        return 1;
    }
    si_vocab *v = NULL;
    if (si_vocab_load(g, &v) != SI_OK) {
        si_log_err("%s", si_errstr());
        si_gguf_close(g);
        return 1;
    }

    int rc = 0;
    if (detok) {
        size_t len;
        char *in = read_all_stdin(&len);
        uint32_t ids[65536];
        size_t n = 0;
        char *end = in + len;
        for (char *p = in; p < end && n < 65536; ) {
            while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == ',')) p++;
            if (p >= end) break;
            char *e;
            long id = strtol(p, &e, 10);
            if (e == p) break;
            ids[n++] = (uint32_t) id;
            p = e;
        }
        char *out = NULL;
        size_t out_len = 0;
        if (si_detokenize(v, ids, n, true, true, &out, &out_len) == SI_OK) {
            fwrite(out, 1, out_len, stdout);
            putchar('\n');
            free(out);
        } else {
            rc = 1;
        }
        free(in);
    } else {
        size_t len;
        char *text;
        if (text_arg) {
            len = strlen(text_arg);
            text = si_xstrdup(text_arg);
        } else {
            text = read_all_stdin(&len);
        }

        uint32_t *ids = NULL;
        size_t n_ids = 0;
        if (si_tokenize(v, text, len, add_special, parse_special, &ids, &n_ids) != SI_OK) {
            si_log_err("%s", si_errstr());
            rc = 1;
        } else {
            if (ids_only) {
                for (size_t i = 0; i < n_ids; i++) {
                    printf("%u%s", ids[i], i + 1 < n_ids ? " " : "");
                }
                putchar('\n');
            } else {
                for (size_t i = 0; i < n_ids; i++) {
                    uint32_t id = ids[i];
                    printf("%6u  ", id);
                    if (id < v->n_tokens) {
                        print_escaped(v->tok_text[id], v->tok_len[id]);
                    }
                    putchar('\n');
                }
            }
            /* roundtrip check */
            char *back = NULL;
            size_t back_len = 0;
            if (rc == 0 && si_detokenize(v, ids, n_ids, true, true, &back, &back_len) == SI_OK) {
                bool same = back_len == len && memcmp(back, text, len) == 0;
                if (!ids_only) {
                    printf("# roundtrip: %s (%zu -> %zu bytes)\n",
                           same ? "ok" : "MISMATCH", len, back_len);
                } else if (!same) {
                    fprintf(stderr, "siphon: roundtrip MISMATCH\n");
                    rc = 1;
                }
                free(back);
            }
            free(ids);
        }
        free(text);
    }

    si_vocab_free(v);
    si_gguf_close(g);
    return rc;
}
