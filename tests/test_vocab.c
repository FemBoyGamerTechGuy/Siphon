/* siphon: tokenizer unit tests with synthetic vocabularies.
 *
 * Builds small SPM and BPE vocabs by hand and checks the merge
 * algorithms, byte fallback, special-token partitioning, bos/eos
 * handling, and detokenization. Real-model fixtures (checked against
 * the reference implementation) live in tests/fixtures/ and run when
 * the model files are present (see README).
 */
#include "siphon/vocab.h"
#include "test_util.h"

#include "../src/tok_internal.h"

/* ---- synthetic vocab construction ------------------------------------ */

static void add_token(si_vocab *v, const char *text, float score, uint8_t type) {
    v->tok_text[v->n_tokens] = si_xstrdup(text);
    v->tok_len[v->n_tokens] = (uint32_t) strlen(text);
    v->tok_score[v->n_tokens] = score;
    v->tok_type[v->n_tokens] = type;
    si_map_put((si_map *) v->str2id, v->tok_text[v->n_tokens],
               v->tok_len[v->n_tokens], v->n_tokens, true);
    v->n_tokens++;
}

static void add_merge(si_vocab *v, const char *l, const char *r) {
    v->merge_l[v->n_merges] = si_xstrdup(l);
    v->merge_llen[v->n_merges] = (uint32_t) strlen(l);
    v->merge_r[v->n_merges] = si_xstrdup(r);
    v->merge_rlen[v->n_merges] = (uint32_t) strlen(r);
    size_t keylen = strlen(l) + 1 + strlen(r);
    char *key = si_xmalloc(keylen + 1);
    memcpy(key, l, strlen(l));
    key[strlen(l)] = '\x01';
    memcpy(key + strlen(l) + 1, r, strlen(r));
    si_map_put((si_map *) v->pair2rank, key, (uint32_t) keylen,
               v->n_merges, false);
    free(key);
    v->n_merges++;
}

static si_vocab *new_vocab(si_vocab_kind kind, size_t cap_tokens, size_t cap_merges) {
    si_vocab *v = si_xcalloc(1, sizeof(*v));
    v->kind = kind;
    v->tok_text = si_xcalloc(cap_tokens, sizeof(char *));
    v->tok_len = si_xcalloc(cap_tokens, sizeof(uint32_t));
    v->tok_score = si_xcalloc(cap_tokens, sizeof(float));
    v->tok_type = si_xcalloc(cap_tokens, sizeof(uint8_t));
    v->str2id = si_xcalloc(1, sizeof(si_map));
    si_map_init((si_map *) v->str2id, cap_tokens);
    v->bos_id = v->eos_id = v->unk_id = -1;
    if (cap_merges) {
        v->merge_l = si_xcalloc(cap_merges, sizeof(char *));
        v->merge_r = si_xcalloc(cap_merges, sizeof(char *));
        v->merge_llen = si_xcalloc(cap_merges, sizeof(uint32_t));
        v->merge_rlen = si_xcalloc(cap_merges, sizeof(uint32_t));
        v->pair2rank = si_xcalloc(1, sizeof(si_map));
        si_map_init((si_map *) v->pair2rank, cap_merges);
    }
    return v;
}

static void free_synth(si_vocab *v) {
    for (uint32_t i = 0; i < v->n_tokens; i++) free((void *) v->tok_text[i]);
    for (uint32_t i = 0; i < v->n_merges; i++) {
        free((void *) v->merge_l[i]);
        free((void *) v->merge_r[i]);
    }
    si_map_free((si_map *) v->str2id); free(v->str2id);
    if (v->pair2rank) { si_map_free((si_map *) v->pair2rank); free(v->pair2rank); }
    free(v->tok_text); free(v->tok_len); free(v->tok_score); free(v->tok_type);
    free(v->merge_l); free(v->merge_r); free(v->merge_llen); free(v->merge_rlen);
    free(v->special_ids);
    free(v);
}

static void finish_specials(si_vocab *v) {
    /* rebuild special cache like the loader does */
    free(v->special_ids);
    v->special_ids = NULL;
    v->n_special = 0;
    uint32_t cap = 16;
    uint32_t *ids = si_xmalloc(cap * sizeof(uint32_t));
    for (uint32_t i = 0; i < v->n_tokens; i++) {
        uint8_t t = v->tok_type[i];
        if (t == SI_TOK_CONTROL || t == SI_TOK_USER || t == SI_TOK_UNKNOWN) {
            if (v->n_special == cap) {
                cap *= 2;
                ids = si_xrealloc(ids, cap * sizeof(uint32_t));
            }
            ids[v->n_special++] = i;
        }
    }
    for (uint32_t i = 1; i < v->n_special; i++) {
        uint32_t x = ids[i], j = i;
        while (j > 0 && (v->tok_len[ids[j-1]] < v->tok_len[x] ||
                         (v->tok_len[ids[j-1]] == v->tok_len[x] && ids[j-1] > x))) {
            ids[j] = ids[j-1];
            j--;
        }
        ids[j] = x;
    }
    v->special_ids = ids;
}

/* ---- SPM tests -------------------------------------------------------- */

static void test_spm(void) {
    si_vocab *v = new_vocab(SI_VOCAB_SPM, 300, 0);
    v->add_bos = true;
    v->add_space_prefix = true;

    add_token(v, "<unk>", 0, SI_TOK_UNKNOWN);   /* 0 */
    add_token(v, "<s>", 0, SI_TOK_CONTROL);     /* 1 */
    add_token(v, "</s>", 0, SI_TOK_CONTROL);    /* 2 */
    add_token(v, "▁hello", 10.0f, SI_TOK_NORMAL); /* 3 */
    add_token(v, "▁world", 9.0f, SI_TOK_NORMAL);  /* 4 */
    add_token(v, "▁w", 6.0f, SI_TOK_NORMAL);      /* 5 */
    add_token(v, "orld", 8.0f, SI_TOK_NORMAL);    /* 6 */
    add_token(v, "or", 5.0f, SI_TOK_NORMAL);      /* 7 */
    add_token(v, "he", 4.0f, SI_TOK_NORMAL);      /* 8 */
    add_token(v, "ll", 3.0f, SI_TOK_NORMAL);      /* 9 */
    add_token(v, "llo", 2.0f, SI_TOK_NORMAL);     /* 10 */
    add_token(v, "ld", 4.0f, SI_TOK_NORMAL);      /* 11 */
    add_token(v, "▁", 1.0f, SI_TOK_NORMAL);       /* 12 */
    add_token(v, "!", 1.0f, SI_TOK_NORMAL);       /* 13 */
    /* byte tokens for x, y, z */
    add_token(v, "<0x78>", 0, SI_TOK_BYTE);       /* 14 */
    add_token(v, "<0x79>", 0, SI_TOK_BYTE);       /* 15 */
    add_token(v, "<0x7A>", 0, SI_TOK_BYTE);       /* 16 */
    /* intermediate merge piece (score drives merge order) */
    add_token(v, "▁he", 5.0f, SI_TOK_NORMAL);     /* 17 */
    v->bos_id = 1;
    v->eos_id = 2;
    v->unk_id = 0;
    for (int b = 0; b < 256; b++) v->byte_tok[b] = -1;
    v->byte_tok[0x78] = 14;
    v->byte_tok[0x79] = 15;
    v->byte_tok[0x7A] = 16;
    finish_specials(v);

    uint32_t *ids = NULL;
    size_t n = 0;

    /* "hello world": dummy prefix + merges should give
     * [bos, ▁hello, ▁world] */
    CHECK(si_tokenize(v, "hello world", 11, true, false, &ids, &n) == SI_OK);
    CHECK(n == 3);
    if (n == 3) CHECK(ids[0] == 1 && ids[1] == 3 && ids[2] == 4);
    free(ids); ids = NULL;

    /* byte fallback for unknown chars */
    CHECK(si_tokenize(v, "xyz", 3, true, false, &ids, &n) == SI_OK);
    CHECK(n == 5);
    if (n == 5) {
        CHECK(ids[0] == 1);          /* bos */
        CHECK(ids[1] == 12);         /* ▁ */
        CHECK(ids[2] == 14 && ids[3] == 15 && ids[4] == 16); /* <0x78..7A> */
    }
    free(ids); ids = NULL;

    /* text already containing a special token: partition, then the
     * following text gets the space prefix again */
    CHECK(si_tokenize(v, "<s>hello", 8, false, true, &ids, &n) == SI_OK);
    CHECK(n == 2);
    if (n == 2) CHECK(ids[0] == 1 && ids[1] == 3);
    free(ids); ids = NULL;

    /* detokenize roundtrip: [bos, ▁hello, ▁world] -> " hello world"
     * (the reference keeps the leading space when the BOS that carried
     * the dummy prefix is stripped) */
    {
        uint32_t in[3] = { 1, 3, 4 };
        char *out = NULL;
        size_t out_len = 0;
        CHECK(si_detokenize(v, in, 3, true, true, &out, &out_len) == SI_OK);
        CHECK(out_len == 12 && memcmp(out, " hello world", 12) == 0);
        free(out);
    }
    /* without bos the first piece's leading space is stripped */
    {
        uint32_t in[2] = { 3, 4 };
        char *out = NULL;
        size_t out_len = 0;
        CHECK(si_detokenize(v, in, 2, true, true, &out, &out_len) == SI_OK);
        CHECK(out_len == 11 && memcmp(out, "hello world", 11) == 0);
        free(out);
    }
    /* without remove_special the control tokens render */
    {
        uint32_t in[3] = { 1, 3, 4 };
        char *out = NULL;
        size_t out_len = 0;
        CHECK(si_detokenize(v, in, 3, false, true, &out, &out_len) == SI_OK);
        CHECK(out_len == 15 && memcmp(out, "<s> hello world", 15) == 0);
        free(out);
    }
    /* control tokens suppressed when unparse_special=false */
    {
        uint32_t in[3] = { 1, 3, 4 };
        char *out = NULL;
        size_t out_len = 0;
        CHECK(si_detokenize(v, in, 3, false, false, &out, &out_len) == SI_OK);
        CHECK(out_len == 12 && memcmp(out, " hello world", 12) == 0);
        free(out);
    }
    /* byte tokens decode to raw bytes */
    {
        uint32_t in[3] = { 14, 15, 16 };
        char *out = NULL;
        size_t out_len = 0;
        CHECK(si_detokenize(v, in, 3, false, true, &out, &out_len) == SI_OK);
        CHECK(out_len == 3 && memcmp(out, "xyz", 3) == 0);
        free(out);
    }

    free_synth(v);
}

/* ---- BPE tests --------------------------------------------------------- */

static void test_bpe(void) {
    si_vocab *v = new_vocab(SI_VOCAB_BPE, 300, 300);
    v->pre = SI_PRE_QWEN2;
    v->clean_spaces = false;

    add_token(v, "a", 0, SI_TOK_NORMAL);
    add_token(v, "b", 0, SI_TOK_NORMAL);
    add_token(v, "c", 0, SI_TOK_NORMAL);
    add_token(v, "1", 0, SI_TOK_NORMAL);
    add_token(v, "2", 0, SI_TOK_NORMAL);
    add_token(v, "3", 0, SI_TOK_NORMAL);
    add_token(v, "ab", 0, SI_TOK_NORMAL);
    add_token(v, "abc", 0, SI_TOK_NORMAL);
    add_token(v, "\xc4\xa0", 0, SI_TOK_NORMAL);        /* Ġ (space symbol) */
    add_token(v, "\xc4\xa0" "ab", 0, SI_TOK_NORMAL);       /* Ġab */
    add_token(v, "\xc4\xa0" "abc", 0, SI_TOK_NORMAL);      /* Ġabc */
    add_token(v, "<|im_start|>", 0, SI_TOK_CONTROL);    /* user-ish */
    add_token(v, "[INST]", 0, SI_TOK_USER);
    uint32_t id_a = 0, id_b = 1, id_abc = 7, id_ghabc = 10, id_ims = 11, id_inst = 12;
    uint32_t id_1 = 3, id_2 = 4, id_3 = 5;

    add_merge(v, "a", "b");                /* rank 0 */
    add_merge(v, "\xc4\xa0", "ab");        /* rank 1: Ġ+ab */
    add_merge(v, "\xc4\xa0" "ab", "c");    /* rank 2: Ġab+c */
    add_merge(v, "ab", "c");               /* rank 3 */
    finish_specials(v);

    uint32_t *ids = NULL;
    size_t n = 0;

    /* "abc" -> single token via merges */
    CHECK(si_tokenize(v, "abc", 3, false, false, &ids, &n) == SI_OK);
    CHECK(n == 1 && ids[0] == id_abc);
    free(ids); ids = NULL;

    /* " abc" -> Ġabc (space joins the word through byte encoding) */
    CHECK(si_tokenize(v, " abc", 4, false, false, &ids, &n) == SI_OK);
    CHECK(n == 1 && ids[0] == id_ghabc);
    free(ids); ids = NULL;

    /* qwen2 pre: digits are single tokens */
    CHECK(si_tokenize(v, "123", 3, false, false, &ids, &n) == SI_OK);
    CHECK(n == 3 && ids[0] == id_1 && ids[1] == id_2 && ids[2] == id_3);
    free(ids); ids = NULL;

    /* "123abc": digits single, letters merge */
    CHECK(si_tokenize(v, "123abc", 6, false, false, &ids, &n) == SI_OK);
    CHECK(n == 4 && ids[0] == id_1 && ids[1] == id_2 && ids[2] == id_3 && ids[3] == id_abc);
    free(ids); ids = NULL;

    /* unmergeable text is dropped when no byte tokens exist (mirrors ref) */
    CHECK(si_tokenize(v, "zzz", 3, false, false, &ids, &n) == SI_OK);
    CHECK(n == 0);
    free(ids); ids = NULL;

    /* special tokens partition the text */
    CHECK(si_tokenize(v, "a<|im_start|>b", 14, false, true, &ids, &n) == SI_OK);
    CHECK(n == 3 && ids[0] == id_a && ids[1] == id_ims && ids[2] == id_b);
    free(ids); ids = NULL;

    /* user-defined tokens partition even without parse_special */
    CHECK(si_tokenize(v, "a[INST]b", 8, false, false, &ids, &n) == SI_OK);
    CHECK(n == 3 && ids[0] == id_a && ids[1] == id_inst && ids[2] == id_b);
    free(ids); ids = NULL;

    /* detokenize: byte-level decode maps Ġ back to space */
    {
        uint32_t in[1] = { id_ghabc };
        char *out = NULL;
        size_t out_len = 0;
        CHECK(si_detokenize(v, in, 1, false, true, &out, &out_len) == SI_OK);
        CHECK(out_len == 4 && memcmp(out, " abc", 4) == 0);
        free(out);
    }
    /* special tokens detokenize to raw text */
    {
        uint32_t in[3] = { id_a, id_ims, id_b };
        char *out = NULL;
        size_t out_len = 0;
        CHECK(si_detokenize(v, in, 3, false, true, &out, &out_len) == SI_OK);
        CHECK(out_len == 14 && memcmp(out, "a<|im_start|>b", 14) == 0);
        free(out);
    }
    /* control tokens suppressed without unparse_special */
    {
        uint32_t in[3] = { id_a, id_ims, id_b };
        char *out = NULL;
        size_t out_len = 0;
        CHECK(si_detokenize(v, in, 3, false, false, &out, &out_len) == SI_OK);
        CHECK(out_len == 2 && memcmp(out, "ab", 2) == 0);
        free(out);
    }

    free_synth(v);
}

/* ---- unicode helpers ---------------------------------------------------- */

static void test_unicode(void) {
    CHECK(si_uni_is_letter('a'));
    CHECK(si_uni_is_letter(0x4E00));      /* CJK */
    CHECK(si_uni_is_letter(0x0430));      /* cyrillic */
    CHECK(!si_uni_is_letter(' '));
    CHECK(si_uni_is_number('3'));
    CHECK(si_uni_is_number(0x0660));      /* arabic-indic digit */
    CHECK(si_uni_is_whitespace(' '));
    CHECK(si_uni_is_whitespace('\n'));
    CHECK(si_uni_is_whitespace(0x3000));  /* ideographic space */
    CHECK(!si_uni_is_whitespace('a'));
    CHECK(!si_uni_is_defined(0x10FFFF));  /* noncharacter edge */
    CHECK(si_uni_is_defined(0x0041));

    /* utf8 decode */
    uint32_t cp;
    CHECK(si_utf8_decode((const uint8_t *) "a", 1, &cp) == 1 && cp == 'a');
    CHECK(si_utf8_decode((const uint8_t *) "\xc3\xa9", 2, &cp) == 2 && cp == 0xE9);
    CHECK(si_utf8_decode((const uint8_t *) "\xe2\x96\x81", 3, &cp) == 3 && cp == 0x2581);
    CHECK(si_utf8_decode((const uint8_t *) "\xf0\x9f\x9a\x80", 4, &cp) == 4 && cp == 0x1F680);
    CHECK(si_utf8_decode((const uint8_t *) "\x80", 1, &cp) == 0);          /* stray cont */
    CHECK(si_utf8_decode((const uint8_t *) "\xc0\x80", 2, &cp) == 0);       /* overlong */
    CHECK(si_utf8_decode((const uint8_t *) "\xe0\x80\x80", 3, &cp) == 0);   /* overlong */
    CHECK(si_utf8_decode((const uint8_t *) "\xf4\x90\x80\x80", 4, &cp) == 0); /* > 10FFFF */

    /* byte-level symbols */
    char sym[3];
    CHECK(si_uni_byte_to_sym('a', sym) == 1 && sym[0] == 'a');
    CHECK(si_uni_byte_to_sym(' ', sym) == 2 && (uint8_t)sym[0] == 0xC4 && (uint8_t)sym[1] == 0xA0);
    CHECK(si_uni_byte_to_sym(0x00, sym) == 2 && (uint8_t)sym[0] == 0xC4 && (uint8_t)sym[1] == 0x80);
    CHECK(si_uni_sym_to_byte("a", 1) == 'a');
    CHECK(si_uni_sym_to_byte("\xc4\xa0", 2) == ' ');
    CHECK(si_uni_sym_to_byte("\xc4\x80", 2) == 0x00);
    CHECK(si_uni_sym_to_byte("z", 1) == -1 ? false : si_uni_sym_to_byte("zz", 2) == -1);

    /* roundtrip all 256 bytes */
    bool ok = true;
    for (int b = 0; b < 256; b++) {
        char s[3];
        size_t k = si_uni_byte_to_sym((uint8_t) b, s);
        if (si_uni_sym_to_byte(s, k) != b) ok = false;
    }
    CHECK(ok);
}

int main(void) {
    test_unicode();
    test_spm();
    test_bpe();
    TEST_REPORT("test_vocab");
}
