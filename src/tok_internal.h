/* siphon: internal tokenizer engine types (private header). */
#ifndef SIPHON_TOK_INTERNAL_H
#define SIPHON_TOK_INTERNAL_H

#include "siphon/common.h"
#include "siphon/unicode.h"
#include "siphon/vocab.h"
#include "map.h"

#include <ctype.h>

/* growable byte buffer */
typedef struct {
    char *d;
    size_t len, cap;
} si_str;

static inline void si_str_push(si_str *s, const void *p, size_t n) {
    if (s->len + n > s->cap) {
        s->cap = (s->len + n) * 2 + 64;
        s->d = si_xrealloc(s->d, s->cap);
    }
    memcpy(s->d + s->len, p, n);
    s->len += n;
}

static inline void si_str_push_byte(si_str *s, char c) {
    si_str_push(s, &c, 1);
}

/* pretokenizer output: byte-encoded words in one buffer */
typedef struct {
    si_str buf;
    size_t *offs;
    size_t *lens;
    size_t n;
} si_words;

int si_pre_split(const char *text, size_t len, si_pre_kind pre, bool byte_encode,
                 si_words *w);
void si_words_free(si_words *w);

/* tokenizer engines (tok_spm.c / tok_bpe.c) */
void si_spm_encode(const si_vocab *v, const char *text, size_t len,
                   uint32_t **ids, size_t *n_ids, size_t *cap_ids);
void si_bpe_encode(const si_vocab *v, const char *text, size_t len,
                   uint32_t **ids, size_t *n_ids, size_t *cap_ids);

static inline void si_push_id(uint32_t **ids, size_t *n, size_t *cap, uint32_t id) {
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 256;
        *ids = si_xrealloc(*ids, *cap * sizeof(uint32_t));
    }
    (*ids)[(*n)++] = id;
}

#endif /* SIPHON_TOK_INTERNAL_H */
