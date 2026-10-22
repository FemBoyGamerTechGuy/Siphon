/* siphon: byte-level BPE tokenizer ("gpt2" models: Qwen2/2.5/3,
 * Llama-3, GPT-2 family).
 *
 * Mirrors the reference implementation:
 *  - pretokenizer regex matchers (gpt2 / qwen2 / llama3 flavors)
 *  - GPT-2 byte-level encoding of each word
 *  - merge ranks drive a priority queue: lowest rank first, ties by
 *    leftmost symbol
 *  - symbols split by UTF-8 lead-byte length
 *  - whole-word shortcut when ignore_merges (llama3)
 *  - leftover symbols fall back to byte tokens
 */
#include "tok_internal.h"
#include "map.h"

typedef struct {
    const char *text;
    size_t n;
    int32_t prev, next;
} bpe_sym;

typedef struct {
    size_t left, right;
    uint32_t rank;
    size_t size;
} bpe_bigram;

/* priority: lower rank first, then smaller left index */
static bool bpe_lower_priority(const bpe_bigram *a, const bpe_bigram *b) {
    if (a->rank != b->rank) return a->rank > b->rank;
    return a->left > b->left;
}

typedef struct {
    bpe_bigram *d;
    size_t n, cap;
} bpe_heap;

static void bpe_heap_push(bpe_heap *h, bpe_bigram b) {
    if (h->n == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 256;
        h->d = si_xrealloc(h->d, h->cap * sizeof(*h->d));
    }
    size_t i = h->n++;
    h->d[i] = b;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!bpe_lower_priority(&h->d[p], &h->d[i])) break;
        bpe_bigram t = h->d[p]; h->d[p] = h->d[i]; h->d[i] = t;
        i = p;
    }
}

static bool bpe_heap_pop(bpe_heap *h, bpe_bigram *out) {
    if (h->n == 0) return false;
    *out = h->d[0];
    h->d[0] = h->d[--h->n];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, best = i;
        if (l < h->n && bpe_lower_priority(&h->d[best], &h->d[l])) best = l;
        if (r < h->n && bpe_lower_priority(&h->d[best], &h->d[r])) best = r;
        if (best == i) break;
        bpe_bigram t = h->d[best]; h->d[best] = h->d[i]; h->d[i] = t;
        i = best;
    }
    return true;
}

typedef struct {
    const si_vocab *v;
    bpe_sym *syms;
    size_t n_syms;
    bpe_heap pq;
} bpe_ctx;

static void bpe_try_add(bpe_ctx *c, int32_t left, int32_t right) {
    if (left < 0 || right < 0 || (size_t) left >= c->n_syms ||
        (size_t) right >= c->n_syms) {
        return;
    }
    bpe_sym *ls = &c->syms[left], *rs = &c->syms[right];
    if (ls->n == 0 || rs->n == 0) return;
    size_t size = ls->n + rs->n;

    /* pair rank lookup: "left\x01right" (0x01 never appears in
     * byte-level encoded text) */
    char key[512];
    if (size + 1 > sizeof(key)) {
        /* oversize pairs can never be merges in practice; skip */
        return;
    }
    memcpy(key, ls->text, ls->n);
    key[ls->n] = '\x01';
    memcpy(key + ls->n + 1, rs->text, rs->n);
    si_map_slot *s = si_map_get((si_map *) c->v->pair2rank, key, (uint32_t)(size + 1));
    if (!s) return;

    bpe_bigram b;
    b.left = (size_t) left;
    b.right = (size_t) right;
    b.rank = (uint32_t) s->val;
    b.size = size;
    bpe_heap_push(&c->pq, b);
}

static void bpe_encode_word(const si_vocab *v, const char *word, size_t len,
                            uint32_t **ids, size_t *n_ids, size_t *cap_ids) {
    bpe_ctx c;
    memset(&c, 0, sizeof(c));
    c.v = v;

    /* whole-word shortcut (llama3 ignore_merges) */
    if (v->ignore_merges) {
        int32_t id = si_vocab_find(v, word, len);
        if (id >= 0) {
            si_push_id(ids, n_ids, cap_ids, (uint32_t) id);
            return;
        }
    }

    /* symbols by UTF-8 lead-byte length */
    size_t offs = 0;
    while (offs < len) {
        size_t k = si_utf8_len_of((uint8_t) word[offs]);
        if (k > len - offs) k = len - offs;
        offs += k;
        c.n_syms++;
    }
    c.syms = si_xcalloc(c.n_syms ? c.n_syms : 1, sizeof(*c.syms));
    offs = 0;
    for (size_t i = 0; i < c.n_syms; i++) {
        size_t k = si_utf8_len_of((uint8_t) word[offs]);
        if (k > len - offs) k = len - offs;
        c.syms[i].text = word + offs;
        c.syms[i].n = k;
        c.syms[i].prev = (int32_t) i - 1;
        c.syms[i].next = (offs + k == len) ? -1 : (int32_t) i + 1;
        offs += k;
    }

    for (size_t i = 1; i < c.n_syms; i++) {
        bpe_try_add(&c, (int32_t) i - 1, (int32_t) i);
    }

    bpe_bigram b;
    while (bpe_heap_pop(&c.pq, &b)) {
        bpe_sym *ls = &c.syms[b.left];
        bpe_sym *rs = &c.syms[b.right];
        if (ls->n == 0 || rs->n == 0 || ls->n + rs->n != b.size) {
            continue;
        }
        ls->n += rs->n;
        rs->n = 0;
        ls->next = rs->next;
        if (rs->next >= 0) {
            c.syms[rs->next].prev = (int32_t) b.left;
        }
        bpe_try_add(&c, ls->prev, (int32_t) b.left);
        bpe_try_add(&c, (int32_t) b.left, ls->next);
    }

    /* emit */
    static const char hex[] = "0123456789ABCDEF";
    for (int32_t i = 0; i != -1; i = c.syms[i].next) {
        if ((size_t) i >= c.n_syms) break;
        bpe_sym *s = &c.syms[i];
        if (s->n == 0) continue;
        int32_t id = si_vocab_find(v, s->text, s->n);
        if (id >= 0) {
            si_push_id(ids, n_ids, cap_ids, (uint32_t) id);
            continue;
        }
        /* byte-level fallback per byte of the symbol */
        for (size_t j = 0; j < s->n; j++) {
            int32_t bid;
            if (v->escape_ws) {
                /* non-byte-encoded BPE (gemma-style): <0xXX> tokens */
                char buf[7] = { '<', '0', 'x',
                                hex[(uint8_t) s->text[j] >> 4],
                                hex[(uint8_t) s->text[j] & 15], '>', 0 };
                bid = si_vocab_find(v, buf, 6);
            } else {
                bid = si_vocab_find(v, s->text + j, 1);
            }
            if (bid >= 0) {
                si_push_id(ids, n_ids, cap_ids, (uint32_t) bid);
            }
        }
    }

    free(c.syms);
    free(c.pq.d);
}

void si_bpe_encode(const si_vocab *v, const char *text, size_t len,
                   uint32_t **ids, size_t *n_ids, size_t *cap_ids) {
    si_words w;
    if (si_pre_split(text, len, v->pre, true, &w) != SI_OK) {
        return;
    }
    for (size_t i = 0; i < w.n; i++) {
        bpe_encode_word(v, w.buf.d + w.offs[i], w.lens[i], ids, n_ids, cap_ids);
    }
    si_words_free(&w);
}
