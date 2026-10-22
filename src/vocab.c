/* siphon: vocabulary loading from GGUF metadata.
 *
 * Loading rules mirror the reference implementation so that the same
 * GGUF produces the same token ids: default special ids per family,
 * GGUF overrides, last-wins duplicate token text, empty tokens renamed
 * to "[EMPTY_i]", merges split at the first space (searching from
 * position 1), special-token cache sorted by text length descending.
 */
#include "siphon/vocab.h"
#include "siphon/unicode.h"
#include "map.h"

#include <ctype.h>

/* ---- arena ------------------------------------------------------------ */

/* NOTE: arena_put must never realloc after the initial allocation —
 * si_vocab_load pre-sizes the arena in a first pass precisely so all
 * returned pointers stay valid (an early version realloc'd lazily and
 * dangled every previously-stored token pointer). */
static char *arena_put(si_vocab *v, const char *s, size_t n) {
    if (v->arena_len + n + 1 > v->arena_cap) {
        /* should be impossible: loader pre-sizes exactly */
        v->arena_cap = (v->arena_len + n + 1) * 2 + 4096;
        v->arena = si_xrealloc(v->arena, v->arena_cap);
        si_log_warn("vocab arena grew unexpectedly; token pointers moved");
    }
    char *p = v->arena + v->arena_len;
    memcpy(p, s, n);
    p[n] = '\0';
    v->arena_len += n + 1;
    return p;
}

/* ---- load ------------------------------------------------------------- */

static int32_t read_id(si_gguf *g, const char *key, int32_t dflt, uint32_t n_tokens) {
    const si_gguf_kv *kv = si_gguf_find(g, key);
    if (!kv) return dflt;
    if (kv->type == SI_GGUF_V_U32 || kv->type == SI_GGUF_V_U64 ||
        kv->type == SI_GGUF_V_I32 || kv->type == SI_GGUF_V_I64) {
        int64_t id = (kv->type == SI_GGUF_V_I32 || kv->type == SI_GGUF_V_I64)
            ? kv->v.i : (int64_t) kv->v.u;
        if (id < 0) return (int32_t) id;  /* explicit -1 (e.g. Qwen bos) */
        if ((uint64_t) id >= n_tokens) {
            si_log_warn("special token '%s' = %lld out of vocab range, using default",
                        key, (long long) id);
            return dflt;
        }
        return (int32_t) id;
    }
    return dflt;
}

int si_vocab_load(si_gguf *g, si_vocab **out) {
    *out = NULL;

    si_gguf_str model = si_gguf_get_str(g, "tokenizer.ggml.model");
    if (!model.ptr) {
        si_set_err("no tokenizer.ggml.model in file (no tokenizer embedded)");
        return SI_EFORMAT;
    }

    si_vocab *v = si_xcalloc(1, sizeof(*v));
    v->bos_id = v->eos_id = v->unk_id = v->pad_id = v->sep_id =
    v->eot_id = v->eom_id = v->linefeed_id = -1;

    if (model.len == 5 && memcmp(model.ptr, "llama", 5) == 0) {
        v->kind = SI_VOCAB_SPM;
        v->add_bos = true;
        v->add_eos = false;
        v->add_space_prefix = true;
        v->clean_spaces = false;
        v->bos_id = 1;
        v->eos_id = 2;
        v->unk_id = 0;
    } else if (model.len == 4 && memcmp(model.ptr, "gpt2", 4) == 0) {
        v->kind = SI_VOCAB_BPE;
        v->add_bos = false;
        v->add_eos = false;
        v->add_space_prefix = false;
        v->clean_spaces = true;
        v->escape_ws = false;
        /* odd but real: reference defaults for gpt2-family */
        v->bos_id = 11;
        v->eos_id = 11;
    } else {
        si_set_err("unsupported tokenizer model '%.*s' (need \"llama\" or \"gpt2\")",
                   (int) model.len, model.ptr);
        free(v);
        return SI_ENOTSUP;
    }

    /* pretokenizer flavor (BPE only) */
    if (v->kind == SI_VOCAB_BPE) {
        si_gguf_str pre = si_gguf_get_str(g, "tokenizer.ggml.pre");
        const char *p = pre.ptr;
        size_t pl = pre.len;
        if (!p || pl == 0) {
            si_log_warn("missing tokenizer.ggml.pre, using 'default' pretokenizer "
                        "(tokenization may differ from the original tokenizer)");
            v->pre = SI_PRE_DEFAULT;
        } else if ((pl == 5 && memcmp(p, "gpt-2", 5) == 0) ||
                   (pl == 5 && memcmp(p, "phi-2", 5) == 0)) {
            v->pre = SI_PRE_GPT2;
        } else if (pl == 5 && memcmp(p, "qwen2", 5) == 0) {
            v->pre = SI_PRE_QWEN2;
            v->clean_spaces = false;
        } else if ((pl == 6 && memcmp(p, "llama3", 6) == 0) ||
                   (pl == 7 && memcmp(p, "llama-v3", 7) == 0) ||
                   (pl == 9 && memcmp(p, "llama-bpe", 9) == 0)) {
            v->pre = SI_PRE_LLAMA3;
            v->ignore_merges = true;
            v->add_bos = true;
        } else {
            /* known-but-unimplemented flavors fall back to gpt2 with a
             * warning rather than failing to load the model at all. */
            si_log_warn("pretokenizer '%.*s' not specifically supported yet, "
                        "using gpt2 rules (tokenization may differ)",
                        (int) pl, p);
            v->pre = SI_PRE_GPT2;
        }
    }

    /* tokens */
    const si_gguf_kv *toks = si_gguf_find_arr(g, "tokenizer.ggml.tokens");
    if (!toks || toks->v.arr.elem != SI_GGUF_V_STRING) {
        si_set_err("tokenizer.ggml.tokens missing or not a string array");
        goto fail;
    }
    v->n_tokens = (uint32_t) toks->v.arr.count;
    v->tok_text = si_xcalloc(v->n_tokens, sizeof(char *));
    v->tok_len = si_xcalloc(v->n_tokens, sizeof(uint32_t));
    v->tok_score = si_xcalloc(v->n_tokens, sizeof(float));
    v->tok_type = si_xcalloc(v->n_tokens, sizeof(uint8_t));

    /* scores: float32 or int32 array, optional */
    const float *scores = NULL;
    const int32_t *iscores = NULL;
    {
        const si_gguf_kv *sk = si_gguf_find_arr(g, "tokenizer.ggml.scores");
        if (sk) {
            if (sk->v.arr.elem == SI_GGUF_V_F32 && sk->v.arr.count >= v->n_tokens) {
                scores = (const float *) sk->v.arr.data;
            } else if (sk->v.arr.elem == SI_GGUF_V_I32 && sk->v.arr.count >= v->n_tokens) {
                iscores = (const int32_t *) sk->v.arr.data;
            } else {
                si_set_err("invalid tokenizer.ggml.scores");
                goto fail;
            }
        }
    }
    /* token types: int32 array, optional */
    const int32_t *toktypes = NULL;
    {
        const si_gguf_kv *tk = si_gguf_find_arr(g, "tokenizer.ggml.token_type");
        if (tk) {
            if (tk->v.arr.elem != SI_GGUF_V_I32 || tk->v.arr.count < v->n_tokens) {
                si_set_err("invalid tokenizer.ggml.token_type");
                goto fail;
            }
            toktypes = (const int32_t *) tk->v.arr.data;
        }
    }

    si_map *str2id = si_xcalloc(1, sizeof(si_map));
    si_map_init(str2id, v->n_tokens + 16);
    v->str2id = str2id;

    /* merges (BPE): sizes needed before the arena can be allocated */
    const si_gguf_kv *mk = NULL;
    if (v->kind == SI_VOCAB_BPE) {
        mk = si_gguf_find_arr(g, "tokenizer.ggml.merges");
        if (!mk || mk->v.arr.elem != SI_GGUF_V_STRING) {
            si_set_err("tokenizer.ggml.merges missing for gpt2 tokenizer");
            goto fail;
        }
        v->n_merges = (uint32_t) mk->v.arr.count;
    }

    /* pass 1: exact arena size (tokens + merges + NULs) */
    {
        size_t need = 0;
        for (uint32_t i = 0; i < v->n_tokens; i++) {
            si_gguf_str t;
            if (si_gguf_arr_str_get(g, "tokenizer.ggml.tokens", i, &t) != SI_OK) {
                si_set_err("failed reading token %u", i);
                goto fail;
            }
            need += t.len ? t.len + 1 : 16;  /* "[EMPTY_nnn]" worst case */
        }
        for (uint32_t i = 0; i < v->n_merges; i++) {
            si_gguf_str m;
            if (si_gguf_arr_str_get(g, "tokenizer.ggml.merges", i, &m) != SI_OK) {
                si_set_err("failed reading merge %u", i);
                goto fail;
            }
            need += m.len + 2;
        }
        v->arena = si_xmalloc(need + 4096);
        v->arena_cap = need + 4096;
        v->arena_len = 0;
    }

    /* pass 2: fill */
    for (uint32_t i = 0; i < v->n_tokens; i++) {
        si_gguf_str t;
        if (si_gguf_arr_str_get(g, "tokenizer.ggml.tokens", i, &t) != SI_OK) {
            si_set_err("failed reading token %u", i);
            goto fail;
        }
        if (t.len == 0) {
            /* reference renames empty tokens so they stay unique */
            char name[32];
            snprintf(name, sizeof(name), "[EMPTY_%u]", i);
            v->tok_text[i] = arena_put(v, name, strlen(name));
            v->tok_len[i] = (uint32_t) strlen(name);
        } else {
            v->tok_text[i] = arena_put(v, t.ptr, t.len);
            v->tok_len[i] = (uint32_t) t.len;
        }
        v->tok_score[i] = scores ? scores[i] : (iscores ? (float) iscores[i] : 0.0f);
        v->tok_type[i] = toktypes ? (uint8_t) (toktypes[i] >= 1 && toktypes[i] <= 6 ?
                                               toktypes[i] : SI_TOK_UNUSED) : SI_TOK_NORMAL;
        /* duplicate text: reference asserts; we warn and let the last
         * assignment win, matching its map assignment order */
        si_map_slot *prev = si_map_get(str2id, v->tok_text[i], v->tok_len[i]);
        if (prev && prev->val != i) {
            si_log_warn("duplicate token text '%s' (ids %llu and %u); using the latter",
                        v->tok_text[i], (unsigned long long) prev->val, i);
        }
        si_map_put(str2id, v->tok_text[i], v->tok_len[i], i, true);
    }

    /* merges (BPE): store split halves in the same arena */
    if (v->kind == SI_VOCAB_BPE) {
        v->merge_l = si_xcalloc(v->n_merges, sizeof(char *));
        v->merge_r = si_xcalloc(v->n_merges, sizeof(char *));
        v->merge_llen = si_xcalloc(v->n_merges, sizeof(uint32_t));
        v->merge_rlen = si_xcalloc(v->n_merges, sizeof(uint32_t));
        si_map *p2r = si_xcalloc(1, sizeof(si_map));
        si_map_init(p2r, v->n_merges + 16);
        v->pair2rank = p2r;
        for (uint32_t i = 0; i < v->n_merges; i++) {
            si_gguf_str m;
            if (si_gguf_arr_str_get(g, "tokenizer.ggml.merges", i, &m) != SI_OK) {
                si_set_err("failed reading merge %u", i);
                goto fail;
            }
            /* split at first space searching from position 1 (reference) */
            size_t pos = 0;
            if (m.len > 1) {
                for (size_t j = 1; j < m.len; j++) {
                    if (m.ptr[j] == ' ') { pos = j; break; }
                }
            }
            if (pos == 0) {
                /* no space: unusable merge entry, leave empty */
                v->merge_l[i] = arena_put(v, "", 0);
                v->merge_r[i] = arena_put(v, "", 0);
                continue;
            }
            v->merge_l[i] = arena_put(v, m.ptr, pos);
            v->merge_llen[i] = (uint32_t) pos;
            v->merge_r[i] = arena_put(v, m.ptr + pos + 1, m.len - pos - 1);
            v->merge_rlen[i] = (uint32_t) (m.len - pos - 1);
        }
        /* rank map: first occurrence wins (reference uses emplace) */
        for (uint32_t i = 0; i < v->n_merges; i++) {
            /* combined key: "l\x01r" in a scratch buffer */
            size_t need = (size_t) v->merge_llen[i] + 1 + v->merge_rlen[i];
            char *key = si_xmalloc(need);
            memcpy(key, v->merge_l[i], v->merge_llen[i]);
            key[v->merge_llen[i]] = '\x01';
            memcpy(key + v->merge_llen[i] + 1, v->merge_r[i], v->merge_rlen[i]);
            si_map_put((si_map *) v->pair2rank, key, (uint32_t) need, i, false);
            free(key);
        }
    }

    /* special ids from GGUF (overrides) */
    v->bos_id = read_id(g, "tokenizer.ggml.bos_token_id", v->bos_id, v->n_tokens);
    v->eos_id = read_id(g, "tokenizer.ggml.eos_token_id", v->eos_id, v->n_tokens);
    v->unk_id = read_id(g, "tokenizer.ggml.unknown_token_id", v->unk_id, v->n_tokens);
    v->pad_id = read_id(g, "tokenizer.ggml.pad_token_id", v->pad_id, v->n_tokens);
    v->sep_id = read_id(g, "tokenizer.ggml.sep_token_id", v->sep_id, v->n_tokens);
    v->eot_id = read_id(g, "tokenizer.ggml.eot_token_id", v->eot_id, v->n_tokens);
    v->eom_id = read_id(g, "tokenizer.ggml.eom_token_id", v->eom_id, v->n_tokens);

    v->add_bos = si_gguf_get_bool(g, "tokenizer.ggml.add_bos_token", v->add_bos);
    v->add_eos = si_gguf_get_bool(g, "tokenizer.ggml.add_eos_token", v->add_eos);
    v->add_space_prefix = si_gguf_get_bool(g, "tokenizer.ggml.add_space_prefix",
                                           v->add_space_prefix);

    /* SPM byte fallback table: <0xXX> tokens, falling back to the raw
     * byte as a token string (both per the reference). */
    if (v->kind == SI_VOCAB_SPM) {
        static const char hex[] = "0123456789ABCDEF";
        for (int b = 0; b < 256; b++) {
            char buf[7] = { '<', '0', 'x', hex[b >> 4], hex[b & 15], '>', 0 };
            si_map_slot *s = si_map_get((si_map *) v->str2id, buf, 6);
            v->byte_tok[b] = s ? (int32_t) s->val : -1;
            if (!s) {
                char one[1] = { (char) b };
                s = si_map_get((si_map *) v->str2id, one, 1);
                v->byte_tok[b] = s ? (int32_t) s->val : -1;
            }
        }
        v->linefeed_id = v->byte_tok['\n'];
    } else {
        /* BPE: byte-level symbols */
        for (int b = 0; b < 256; b++) {
            char sym[3];
            size_t sl = si_uni_byte_to_sym((uint8_t) b, sym);
            si_map_slot *s = si_map_get((si_map *) v->str2id, sym, (uint32_t) sl);
            v->byte_tok[b] = s ? (int32_t) s->val : -1;
        }
        v->linefeed_id = v->byte_tok['\n'];
    }

    /* Token texts that the reference promotes to CONTROL (its EOG list:
     * models like Qwen ship "</s>" as a plain normal token, but the
     * reference partitions on it when parse_special is set). Empty-token
     * entries never match because empty tokens were renamed above. */
    {
        static const char *const eog_texts[] = {
            "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>",
            "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>", "",
            "<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]",
            "<eos>", "<turn|>", "<|tool_response>", "<|end_of_utterance>",
            "</s>", "<|eom_id|>",
            "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">",
            "[e~[", "<|ifm|im_end|>",
        };
        for (uint32_t i = 0; i < v->n_tokens; i++) {
            if (v->tok_type[i] == SI_TOK_CONTROL) continue;
            for (size_t e = 0; e < sizeof(eog_texts) / sizeof(eog_texts[0]); e++) {
                if (strcmp(v->tok_text[i], eog_texts[e]) == 0) {
                    si_log_warn("token %u '%s' promoted to control (EOG heuristic)",
                                i, v->tok_text[i]);
                    v->tok_type[i] = SI_TOK_CONTROL;
                    break;
                }
            }
        }
    }

    /* special-token cache: CONTROL | USER | UNKNOWN, length desc */
    {
        uint32_t cap = 16;
        uint32_t n = 0;
        uint32_t *ids = si_xmalloc(cap * sizeof(uint32_t));
        for (uint32_t i = 0; i < v->n_tokens; i++) {
            uint8_t t = v->tok_type[i];
            if (t == SI_TOK_CONTROL || t == SI_TOK_USER || t == SI_TOK_UNKNOWN) {
                if (n == cap) { cap *= 2; ids = si_xrealloc(ids, cap * sizeof(uint32_t)); }
                ids[n++] = i;
            }
        }
        /* insertion sort by length desc, id asc for determinism */
        for (uint32_t i = 1; i < n; i++) {
            uint32_t x = ids[i];
            uint32_t j = i;
            while (j > 0) {
                uint32_t y = ids[j - 1];
                bool swap = v->tok_len[y] < v->tok_len[x] ||
                            (v->tok_len[y] == v->tok_len[x] && y > x);
                if (!swap) break;
                ids[j] = ids[j - 1];
                j--;
            }
            ids[j] = x;
        }
        v->special_ids = ids;
        v->n_special = n;
    }

    *out = v;
    return SI_OK;

fail:
    si_vocab_free(v);
    return SI_EFORMAT;
}

void si_vocab_free(si_vocab *v) {
    if (!v) return;
    if (v->str2id) { si_map_free((si_map *) v->str2id); free(v->str2id); }
    if (v->pair2rank) { si_map_free((si_map *) v->pair2rank); free(v->pair2rank); }
    free(v->tok_text);
    free(v->tok_len);
    free(v->tok_score);
    free(v->tok_type);
    free(v->merge_l);
    free(v->merge_r);
    free(v->merge_llen);
    free(v->merge_rlen);
    free(v->special_ids);
    free(v->arena);
    free(v);
}

int32_t si_vocab_find(const si_vocab *v, const char *s, size_t len) {
    si_map_slot *sl = si_map_get((si_map *) v->str2id, s, (uint32_t) len);
    return sl ? (int32_t) sl->val : -1;
}
