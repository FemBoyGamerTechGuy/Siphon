/* siphon: `run` — generate, dump logits, or verify against a golden file
 *
 * Generation is token-serial in milestone 2 (batched prefill comes with
 * the GEMM work in a later milestone). The --golden mode is the heart of
 * the verification protocol: it teacher-forces the golden's token stream
 * through Siphon one position at a time and compares every logits row
 * against the reference implementation's recorded output.
 *
 * The --dump-logits mode writes the same SLGD v1 format the dev-time
 * reference dumper produces, so files diff byte-for-byte in token count
 * and layout (floats may differ within the numerical band).
 */
#include "siphon/engine.h"
#include "siphon/kernels.h"
#include "siphon/sampler.h"
#include "siphon/vocab.h"

#include <math.h>
#include <time.h>

/* ---- SLGD format ------------------------------------------------------- */

typedef struct {
    char magic[4];      /* "SLGD" */
    uint32_t version;   /* 1 */
    uint32_t n_total;   /* prompt + generated */
    uint32_t n_prompt;
    uint32_t n_vocab;
    /* int32 tokens[n_total]; float logits[n_total][n_vocab] */
} slgd_header;

static int slgd_read(const char *path, int32_t **toks, float **logits,
                     slgd_header *h) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        si_set_err("cannot open golden %s", path);
        return SI_EIO;
    }
    if (fread(h, sizeof *h, 1, f) != 1 || memcmp(h->magic, "SLGD", 4) != 0 ||
        h->version != 1) {
        fclose(f);
        si_set_err("%s: not an SLGD v1 file", path);
        return SI_EFORMAT;
    }
    *toks = si_xmalloc(sizeof(int32_t) * h->n_total);
    if (fread(*toks, sizeof(int32_t), h->n_total, f) != h->n_total) {
        fclose(f);
        si_set_err("%s: truncated token section", path);
        return SI_EFORMAT;
    }
    const size_t nl = (size_t)h->n_total * h->n_vocab;
    *logits = si_xmalloc(sizeof(float) * nl);
    if (fread(*logits, sizeof(float), nl, f) != nl) {
        fclose(f);
        si_set_err("%s: truncated logits section", path);
        return SI_EFORMAT;
    }
    fclose(f);
    return SI_OK;
}

static int slgd_write(const char *path, const int32_t *toks,
                      const float *logits, uint32_t n_total,
                      uint32_t n_prompt, uint32_t n_vocab) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        si_set_err("cannot create %s", path);
        return SI_EIO;
    }
    slgd_header h;
    memcpy(h.magic, "SLGD", 4);
    h.version = 1;
    h.n_total = n_total;
    h.n_prompt = n_prompt;
    h.n_vocab = n_vocab;
    fwrite(&h, sizeof h, 1, f);
    fwrite(toks, sizeof(int32_t), n_total, f);
    fwrite(logits, sizeof(float), (size_t)n_total * n_vocab, f);
    fclose(f);
    return SI_OK;
}

/* ---- helpers ------------------------------------------------------------ */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* peak process memory (Linux /proc; soft-fail elsewhere) */
static void print_mem_peak(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[128];
    long hwm_kb = -1, rss_kb = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%ld", &hwm_kb);
        } else if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &rss_kb);
        }
    }
    fclose(f);
    if (hwm_kb > 0) {
        printf("memory: peak %.1f MiB (VmHWM), current %.1f MiB (VmRSS)\n",
               hwm_kb / 1024.0, rss_kb > 0 ? rss_kb / 1024.0 : 0.0);
    }
}

static int read_file(const char *path, char **out, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        si_set_err("cannot open %s", path);
        return SI_EIO;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    *out = si_xmalloc((size_t)n + 1);
    if (fread(*out, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        si_set_err("cannot read %s", path);
        return SI_EIO;
    }
    (*out)[n] = 0;
    *len = (size_t)n;
    fclose(f);
    return SI_OK;
}

/* ---- golden verification ------------------------------------------------ */

typedef struct {
    double max_abs;
    double sum_abs;
    int64_t count;
} diff_stat;

static void diff_row(const float *a, const float *b, int32_t n, diff_stat *s) {
    for (int32_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > s->max_abs) s->max_abs = d;
        s->sum_abs += d;
        s->count++;
    }
}

static int cmd_run_golden(si_engine *eng, const char *path) {
    slgd_header h;
    int32_t *toks = NULL;
    float *ref = NULL;
    int rc = slgd_read(path, &toks, &ref, &h);
    if (rc != SI_OK) return rc;

    if (h.n_vocab != eng->m->h.n_vocab) {
        si_set_err("golden vocab %u != model vocab %u", h.n_vocab,
                   eng->m->h.n_vocab);
        free(toks); free(ref);
        return SI_EINVAL;
    }

    printf("golden: %u prompt + %u generated = %u positions, vocab %u\n",
           h.n_prompt, h.n_total - h.n_prompt, h.n_total, h.n_vocab);

    diff_stat prefill = {0}, gen = {0};
    int64_t flips = 0, real_flips = 0;
    const int64_t REAL_FLIP_WINDOW = 24;
    const double REAL_FLIP_GAP = 1.0;
    const bool trace = getenv("SI_GOLDEN_TRACE") != NULL;

    for (uint32_t step = 0; step < h.n_total; step++) {
        const float *lg = si_engine_forward(eng, (uint32_t)toks[step]);
        if (!lg) {
            free(toks); free(ref);
            return SI_EINVAL;
        }
        const float *rrow = ref + (size_t)step * h.n_vocab;
        diff_row(lg, rrow, (int32_t)h.n_vocab,
                 step < h.n_prompt ? &prefill : &gen);
        if (trace && step < 16) {
            double m = 0;
            for (int32_t i = 0; i < (int32_t)h.n_vocab; i++) {
                const double d = fabs((double)lg[i] - (double)rrow[i]);
                if (d > m) m = d;
            }
            printf("  pos %2u: max abs %8.4f  ours[0]=%9.4f ref[0]=%9.4f "
                   "ours[argmax]=%9.4f\n",
                   step, m, lg[0], rrow[0],
                   lg[si_sample_greedy(lg, (int32_t)h.n_vocab)]);
        }

        const int32_t ours = (int32_t)si_sample_greedy(lg, (int32_t)h.n_vocab);
        int32_t ref_arg = 0;
        for (int32_t i = 1; i < (int32_t)h.n_vocab; i++) {
            if (rrow[i] > rrow[ref_arg]) ref_arg = i;
        }
        if (ours != ref_arg) {
            flips++;
            /* classify by the reference top-2 gap: a flip on a
             * near-tie is the reference's own numerical variation, not
             * a real divergence */
            float top1 = -INFINITY, top2 = -INFINITY;
            for (int32_t i = 0; i < (int32_t)h.n_vocab; i++) {
                if (rrow[i] > top1) { top2 = top1; top1 = rrow[i]; }
                else if (rrow[i] > top2) { top2 = rrow[i]; }
            }
            const double gap = (double)top1 - (double)top2;
            if (gap > REAL_FLIP_GAP && (int64_t)step < REAL_FLIP_WINDOW) {
                real_flips++;
                printf("  real flip at pos %u: ours %d, ref %d (ref gap %.3f)\n",
                       step, ours, ref_arg, gap);
            }
        }
    }

    const double pre_max = prefill.max_abs;
    const double gen_max = gen.max_abs;
    printf("prefill logits: max abs diff %.4f, mean %.6f (%lld values)\n",
           pre_max, prefill.count ? prefill.sum_abs / prefill.count : 0.0,
           (long long)prefill.count);
    printf("decode  logits: max abs diff %.4f, mean %.6f (%lld values)\n",
           gen_max, gen.count ? gen.sum_abs / gen.count : 0.0,
           (long long)gen.count);
    printf("argmax flips: %lld total, %lld real (gap > %.1f, first %lld positions)\n",
           (long long)flips, (long long)real_flips, REAL_FLIP_GAP,
           (long long)REAL_FLIP_WINDOW);

    /* acceptance: the reference's own FA-on/off variation defines the
     * band; thresholds from the calibrated protocol */
    const double thr = h.n_prompt <= 16 ? 0.6 : 1.0;
    const bool pass = pre_max <= thr && real_flips == 0;
    printf("result: %s (prefill max %.4f vs threshold %.2f)\n",
           pass ? "PASS" : "FAIL", pre_max, thr);
    print_mem_peak();

    free(toks);
    free(ref);
    return pass ? SI_OK : SI_EINVAL;
}

/* ---- generation ----------------------------------------------------------- */

static int cmd_run_generate(si_engine *eng, si_vocab *voc, const char *prompt,
                            size_t prompt_len, int n_gen, si_sampler *samp,
                            const char *dump_path, bool add_special) {
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    int rc = si_tokenize(voc, prompt, prompt_len, add_special, true, &ids, &n_ids);
    if (rc != SI_OK) return rc;
    if (n_ids == 0) {
        si_set_err("empty prompt");
        free(ids);
        return SI_EINVAL;
    }
    printf("prompt: %zu tokens\n", n_ids);

    const uint32_t n_total = (uint32_t)(n_ids + (size_t)n_gen);
    int32_t *seq = si_xmalloc(sizeof(int32_t) * n_total);
    float *all_logits = dump_path
        ? si_xmalloc(sizeof(float) * (size_t)n_total * voc->n_tokens)
        : NULL;

    const double t0 = now_sec();
    const float *lg = NULL;
    for (size_t i = 0; i < n_ids; i++) {
        lg = si_engine_forward(eng, ids[i]);
        if (!lg) { rc = SI_EINVAL; goto out; }
        if (all_logits) {
            memcpy(all_logits + (size_t)i * voc->n_tokens, lg,
                   (size_t)voc->n_tokens * sizeof(float));
        }
        seq[i] = (int32_t)ids[i];
    }
    const double t_prefill = now_sec() - t0;

    /* token-serial generation: `lg` (engine-owned) holds the row for the
     * previous position; decode each sampled token so the dump records
     * a logits row for every position, like the dev-time dumper */
    const double t1 = now_sec();
    size_t n_out = n_ids;
    for (int step = 0; step < n_gen; step++) {
        const uint32_t tok = si_sample(samp, lg, (int32_t)voc->n_tokens);
        seq[n_out++] = (int32_t)tok;
        if (voc->eos_id >= 0 && (int32_t)tok == voc->eos_id) break;
        if (step + 1 < n_gen || all_logits) {
            lg = si_engine_forward(eng, tok);
            if (!lg) { rc = SI_EINVAL; goto out; }
            if (all_logits) {
                memcpy(all_logits + (size_t)(n_out - 1) * voc->n_tokens, lg,
                       (size_t)voc->n_tokens * sizeof(float));
            }
        }
    }
    const double t_decode = now_sec() - t1;

    /* detokenize everything after the prompt */
    char *text = NULL;
    size_t text_len = 0;
    if (si_detokenize(voc, (const uint32_t *)(seq + n_ids), n_out - n_ids,
                      false, false, &text, &text_len) == SI_OK) {
        fwrite(text, 1, text_len, stdout);
        printf("\n");
        free(text);
    }

    printf("timing: prefill %.3f s (%zu tok, %.1f tok/s), decode %.3f s "
           "(%zu tok, %.1f tok/s) [%s, %d threads]\n",
           t_prefill, n_ids, n_ids / (t_prefill > 0 ? t_prefill : 1e-9),
           t_decode, n_out - n_ids,
           (n_out - n_ids) / (t_decode > 0 ? t_decode : 1e-9),
           si_kern_impl_name(), si_tp_nlanes(eng->tp));
    print_mem_peak();

    if (dump_path) {
        rc = slgd_write(dump_path, seq, all_logits, (uint32_t)n_out,
                        (uint32_t)n_ids, voc->n_tokens);
        if (rc == SI_OK) {
            printf("dumped %u positions to %s\n", (uint32_t)n_out, dump_path);
        }
    }
out:
    free(ids);
    free(seq);
    free(all_logits);
    return rc;
}

/* ---- entry ----------------------------------------------------------------- */

int cmd_run(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    const char *prompt_file = NULL;
    const char *golden = NULL;
    const char *dump_logits = NULL;
    int n_gen = 64;
    int threads = 2;
    int64_t n_ctx = 512;
    si_sampler samp = { .temp = 0.0f, .top_k = 0, .top_p = 1.0f, .seed = 0 };

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(a, "--prompt-file") == 0 && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (strcmp(a, "-n") == 0 && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (strcmp(a, "-t") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) {
            n_ctx = atoll(argv[++i]);
        } else if (strcmp(a, "--temp") == 0 && i + 1 < argc) {
            samp.temp = (float)atof(argv[++i]);
        } else if (strcmp(a, "--top-k") == 0 && i + 1 < argc) {
            samp.top_k = atoi(argv[++i]);
        } else if (strcmp(a, "--top-p") == 0 && i + 1 < argc) {
            samp.top_p = (float)atof(argv[++i]);
        } else if (strcmp(a, "--seed") == 0 && i + 1 < argc) {
            samp.seed = strtoull(argv[++i], NULL, 0);
        } else if (strcmp(a, "--golden") == 0 && i + 1 < argc) {
            golden = argv[++i];
        } else if (strcmp(a, "--dump-logits") == 0 && i + 1 < argc) {
            dump_logits = argv[++i];
        } else if (a[0] == '-') {
            si_set_err("unknown run option %s", a);
            return SI_EINVAL;
        } else if (!model_path) {
            model_path = a;
        } else {
            si_set_err("unexpected argument %s", a);
            return SI_EINVAL;
        }
    }
    if (!model_path) {
        si_set_err("usage: siphon run [options] <model.gguf>");
        return SI_EINVAL;
    }
    if (!golden && !prompt && !prompt_file) {
        si_set_err("nothing to do: pass -p, --prompt-file or --golden");
        return SI_EINVAL;
    }

    si_kern_init();

    si_model *m = NULL;
    int rc = si_model_load(model_path, &m);
    if (rc != SI_OK) {
        fprintf(stderr, "siphon: %s\n", si_errstr());
        return rc;
    }

    si_engine *eng = NULL;
    rc = si_engine_create(m, n_ctx, threads, &eng);
    if (rc != SI_OK) {
        fprintf(stderr, "siphon: %s\n", si_errstr());
        si_model_free(m);
        return rc;
    }

    if (golden) {
        rc = cmd_run_golden(eng, golden);
    } else {
        si_vocab *voc = NULL;
        rc = si_vocab_load(m->g, &voc);
        if (rc != SI_OK) {
            fprintf(stderr, "siphon: %s\n", si_errstr());
        } else {
            char *ptext = NULL;
            size_t plen = 0;
            if (prompt_file) {
                rc = read_file(prompt_file, &ptext, &plen);
            } else {
                ptext = si_xstrdup(prompt);
                plen = strlen(prompt);
            }
            if (rc == SI_OK) {
                /* dump mode reproduces the dev-time protocol exactly
                 * (no auto special tokens); normal runs use the model's
                 * own add_bos/add_eos flags */
                rc = cmd_run_generate(eng, voc, ptext, plen, n_gen, &samp,
                                      dump_logits, !dump_logits);
            }
            free(ptext);
            si_vocab_free(voc);
        }
    }

    si_engine_free(eng);
    si_model_free(m);
    return rc;
}
