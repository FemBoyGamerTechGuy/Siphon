/* siphon: tokenizer vocabulary loaded from GGUF metadata.
 *
 * Two tokenizer families are supported, matching what real models ship:
 *   - "llama" (SentencePiece BPE): score-priority merges over the
 *     space-escaped text, byte fallback via <0xXX> tokens.
 *   - "gpt2" (byte-level BPE): regex pretokenizer + rank-priority
 *     merges over GPT-2 byte-encoded words.
 *
 * The algorithms mirror the reference implementation (llama.cpp,
 * dev-time reference only — never linked) including its edge-case
 * behavior, so token ids match exactly on real models.
 */
#ifndef SIPHON_VOCAB_H
#define SIPHON_VOCAB_H

#include "siphon/common.h"
#include "siphon/gguf.h"

/* token types, GGUF numbering */
typedef enum {
    SI_TOK_NORMAL   = 1,
    SI_TOK_UNKNOWN  = 2,
    SI_TOK_CONTROL  = 3,
    SI_TOK_USER     = 4,
    SI_TOK_UNUSED   = 5,
    SI_TOK_BYTE     = 6,
} si_tok_type;

typedef enum {
    SI_VOCAB_SPM = 0,   /* tokenizer.ggml.model == "llama" */
    SI_VOCAB_BPE = 1,   /* tokenizer.ggml.model == "gpt2"   */
} si_vocab_kind;

/* pretokenizer regex flavor (tokenizer.ggml.pre) */
typedef enum {
    SI_PRE_DEFAULT = 0, /* unknown/missing pre: the reference's fallback */
    SI_PRE_GPT2    = 1, /* "gpt-2" and friends */
    SI_PRE_QWEN2   = 2, /* "qwen2" (Qwen2/2.5/3) */
    SI_PRE_LLAMA3  = 3, /* "llama3" and friends */
} si_pre_kind;

typedef struct {
    si_vocab_kind kind;
    si_pre_kind pre;

    /* behavior flags (defaults per family, overridden by GGUF keys) */
    bool add_bos;            /* tokenizer.ggml.add_bos_token  */
    bool add_eos;            /* tokenizer.ggml.add_eos_token  */
    bool add_space_prefix;   /* tokenizer.ggml.add_space_prefix */
    bool clean_spaces;       /* detokenize whitespace cleanup  */
    bool ignore_merges;      /* whole-word lookup shortcut (llama3) */
    bool escape_ws;          /* BPE with SPM-style ▁ escaping (gemma; off for gpt2/qwen) */

    /* tokens (text lives in an owned arena; vocab is self-contained) */
    uint32_t n_tokens;
    const char **tok_text;   /* nul-terminated, arena pointers */
    uint32_t *tok_len;
    float *tok_score;
    uint8_t *tok_type;       /* si_tok_type */

    /* merges (BPE only): rank i = merge_l[i] + " " + merge_r[i] */
    uint32_t n_merges;
    const char **merge_l, **merge_r;
    uint32_t *merge_llen, *merge_rlen;

    /* special ids, -1 when absent */
    int32_t bos_id, eos_id, unk_id, pad_id, sep_id, eot_id, eom_id;
    int32_t linefeed_id;

    /* lookups */
    void *str2id;            /* token text -> id (si_map, opaque here) */
    void *pair2rank;         /* merge pair -> rank (si_map) */

    /* special tokens for text partitioning, sorted by length desc */
    uint32_t *special_ids;
    uint32_t n_special;

    /* SPM byte fallback: byte value -> token id (-1 if none) */
    int32_t byte_tok[256];

    char *arena;             /* owned token/merge strings */
    size_t arena_len, arena_cap;
} si_vocab;

/* Load the vocabulary from an open GGUF. The si_gguf must outlive the
 * call (only used during load; strings are copied). */
int si_vocab_load(si_gguf *g, si_vocab **out);
void si_vocab_free(si_vocab *v);

/* Tokenize `text` (arbitrary bytes, len may include NULs).
 * add_special: prepend/append bos/eos per the model's flags.
 * parse_special: recognize special tokens in the text (e.g. <|im_end|>);
 *   user-defined tokens are always recognized, like the reference.
 * Returns SI_OK with *ids (malloc'd, caller frees) or negative error. */
int si_tokenize(const si_vocab *v, const char *text, size_t len,
                bool add_special, bool parse_special,
                uint32_t **ids, size_t *n_ids);

/* Detokenize. remove_special drops leading bos / trailing eos.
 * unparse_special: control/unknown tokens render as their text;
 *   otherwise they are suppressed (empty), like the reference.
 * Returns SI_OK with *text (malloc'd, NUL-terminated, *len excludes
 * the terminator) or negative error. */
int si_detokenize(const si_vocab *v, const uint32_t *ids, size_t n_ids,
                  bool remove_special, bool unparse_special,
                  char **text, size_t *len);

/* Look up a token id by exact text; -1 if absent. */
int32_t si_vocab_find(const si_vocab *v, const char *s, size_t len);

#endif /* SIPHON_VOCAB_H */
