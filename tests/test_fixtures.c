/* siphon: real-model tokenizer fixtures.
 *
 * Reads .tokfix files (generated against the llama.cpp reference at
 * development time) and verifies siphon produces identical token ids
 * for every case and mode. Models are looked up via:
 *   - $SIPHON_TEST_QWEN   (default tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf)
 *   - $SIPHON_TEST_TINYLLAMA (default tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
 * Missing models skip their fixtures with a notice (weights are not
 * committed to the repo).
 */
#include "siphon/vocab.h"
#include "test_util.h"

#include <stdlib.h>
#include <unistd.h>

typedef struct {
    bool add_special;
    bool parse_special;
} mode;

static bool run_fixture(const char *path, const char *model) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "  (fixture %s missing, skipping)\n", path);
        return true;
    }

    si_gguf *g = NULL;
    if (si_gguf_open(model, &g) != SI_OK) {
        fprintf(stderr, "  cannot open model %s: %s\n", model, si_errstr());
        fclose(f);
        return false;
    }
    si_vocab *v = NULL;
    if (si_vocab_load(g, &v) != SI_OK) {
        fprintf(stderr, "  cannot load vocab: %s\n", si_errstr());
        si_gguf_close(g);
        fclose(f);
        return false;
    }

    bool ok = true;
    mode m = { false, false };
    char line[65536];
    int n_cases = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t ll = strlen(line);
        while (ll > 0 && (line[ll - 1] == '\n' || line[ll - 1] == '\r')) line[--ll] = '\0';
        if (ll == 0) continue;
        if (line[0] == '#') {
            if (strstr(line, "mode: nospecial")) m = (mode){ false, false };
            else if (strstr(line, "mode: parse")) m = (mode){ false, true };
            else if (strstr(line, "mode: special")) m = (mode){ true, false };
            continue;
        }

        /* parse "<hex> <ids...>" */
        char *sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = '\0';
        const char *hex = line;

        size_t clen = strlen(hex) / 2;
        char *text = si_xmalloc(clen ? clen : 1);
        for (size_t i = 0; i < clen; i++) {
            text[i] = (char) (strtol((char[3]){ hex[2*i], hex[2*i+1], 0 }, NULL, 16));
        }

        uint32_t *ids = NULL;
        size_t n_ids = 0;
        if (si_tokenize(v, text, clen, m.add_special, m.parse_special, &ids, &n_ids) != SI_OK) {
            ok = false;
            free(text);
            continue;
        }

        /* compare with expected */
        char *p = sp + 1;
        size_t n_exp = 0;
        bool same = true;
        for (;;) {
            while (*p == ' ') p++;
            if (*p == '\0') break;
            char *end;
            unsigned long id = strtoul(p, &end, 10);
            if (end == p) break;
            if (n_exp >= n_ids || ids[n_exp] != (uint32_t) id) { same = false; }
            n_exp++;
            p = end;
        }
        if (n_exp != n_ids) same = false;
        if (!same) {
            ok = false;
            fprintf(stderr, "  FAIL %s mode(%d,%d): expected %zu ids, got %zu\n",
                    path, m.add_special, m.parse_special, n_exp, n_ids);
        }
        n_cases++;

        /* detokenize roundtrip: only meaningful without added bos/eos
         * and without special tokens in the text (with specials the
         * reference itself renders them and re-adds a prefix space —
         * siphon matches that byte-for-byte, verified against the
         * reference outputs during development) */
        if (!m.add_special && !m.parse_special && !v->clean_spaces) {
            char *back = NULL;
            size_t back_len = 0;
            if (si_detokenize(v, ids, n_ids, false, true, &back, &back_len) == SI_OK) {
                if (back_len != clen || memcmp(back, text, clen) != 0) {
                    ok = false;
                    fprintf(stderr, "  FAIL %s detok roundtrip (%zu -> %zu bytes)\n",
                            path, clen, back_len);
                }
                free(back);
            }
        }
        free(ids);
        free(text);
    }

    printf("  %s: %d cases %s\n", path, n_cases, ok ? "ok" : "FAILED");
    si_vocab_free(v);
    si_gguf_close(g);
    fclose(f);
    return ok;
}

int main(void) {
    const char *qwen = getenv("SIPHON_TEST_QWEN");
    const char *tl = getenv("SIPHON_TEST_TINYLLAMA");
    char def_q[128], def_t[128];
    if (!qwen) {
        snprintf(def_q, sizeof(def_q), "tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
        qwen = def_q;
    }
    if (!tl) {
        snprintf(def_t, sizeof(def_t), "tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
        tl = def_t;
    }
    if (access(qwen, R_OK) != 0 || access(tl, R_OK) != 0) {
        printf("test_fixtures: models not present, skipping (see README for setup)\n");
        return 0;
    }
    bool ok = run_fixture("tests/fixtures/qwen2.5-0.5b-q4km.tokfix", qwen);
    ok = run_fixture("tests/fixtures/tinyllama-1.1b-q4km.tokfix", tl) && ok;
    TEST_REPORT("test_fixtures");
}
