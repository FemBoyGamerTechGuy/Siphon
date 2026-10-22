/* siphon: SentencePiece-family tokenizer ("llama" models).
 *
 * Mirrors the reference implementation exactly:
 *  - text is space-escaped (' ' -> U+2581) by the caller
 *  - symbols split by UTF-8 lead byte length (invalid bytes join the
 *    following bytes into one symbol, like the reference's len table)
 *  - BPE merges driven by token scores via a priority queue:
 *    highest score first, ties broken by leftmost symbol
 *  - unmergeable leftovers fall back to <0xXX> byte tokens
 */
#include "tok_internal.h"
#include "map.h"

typedef struct {
    const char *text;   /* into the (escaped) input */
    size_t n;           /* byte length; 0 when merged away */
    int32_t prev, next;
} spm_sym;

typedef struct {
    size_t left, right;
    float score;
    size_t size;
} spm_bigram;

/* priority: higher score first, then smaller left index */
static bool spm_lower_priority(const spm_bigram *a, const spm_bigram *b) {
    if (a->score != b->score) return a->score < b->score;
    return a->left > b->left;
}

typedef struct {
    spm_bigram *d;
    size_t n, cap;
} spm_heap;

static void spm_heap_push(spm_heap *h, spm_bigram b) {
    if (h->n == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 256;
        h->d = si_xrealloc(h->d, h->cap * sizeof(*h->d));
    }
    size_t i = h->n++;
    h->d[i] = b;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!spm_lower_priority(&h->d[p], &h->d[i])) break;
        spm_bigram t = h->d[p]; h->d[p] = h->d[i]; h->d[i] = t;
        i = p;
    }
}

static bool spm_heap_pop(spm_heap *h, spm_bigram *out) {
    if (h->n == 0) return false;
    *out = h->d[0];
    h->d[0] = h->d[--h->n];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, best = i;
        if (l < h->n && spm_lower_priority(&h->d[best], &h->d[l])) best = l;
        if (r < h->n && spm_lower_priority(&h->d[best], &h->d[r])) best = r;
        if (best == i) break;
        spm_bigram t = h->d[best]; h->d[best] = h->d[i]; h->d[i] = t;
        i = best;
    }
    return true;
}

typedef struct {
    const si_vocab *v;
    spm_sym *syms;
    size_t n_syms;
    spm_heap pq;
    si_map rev_merge;   /* merged text -> (left<<32 | right) */
    bool warned_no_byte_tok;
} spm_ctx;

static void spm_try_add(spm_ctx *c, int32_t left, int32_t right) {
    if (left < 0 || right < 0 || (size_t) left >= c->n_syms ||
        (size_t) right >= c->n_syms) {
        return;
    }
    const char *lt = c->syms[left].text;
    size_t ln = c->syms[left].n, rn = c->syms[right].n;
    if (ln == 0 || rn == 0) return;
    size_t size = ln + rn;

    int32_t id = si_vocab_find(c->v, lt, size);
    if (id < 0) return;

    spm_bigram b;
    b.left = (size_t) left;
    b.right = (size_t) right;
    b.score = c->v->tok_score[id];
    b.size = size;
    spm_heap_push(&c->pq, b);

    /* rev_merge records how to re-segment a merged span; later writes
     * win, matching the reference's map assignment */
    si_map_put(&c->rev_merge, lt, (uint32_t) size,
               ((uint64_t)(uint32_t) left << 32) | (uint32_t) right, true);
}

static void spm_resegment(spm_ctx *c, size_t i, uint32_t **ids, size_t *n_ids,
                          size_t *cap_ids) {
    const spm_sym *s = &c->syms[i];
    if (s->n == 0) return;

    int32_t id = si_vocab_find(c->v, s->text, s->n);
    if (id >= 0) {
        si_push_id(ids, n_ids, cap_ids, (uint32_t) id);
        return;
    }

    si_map_slot *rev = si_map_get(&c->rev_merge, s->text, (uint32_t) s->n);
    if (rev) {
        spm_resegment(c, (size_t)(uint32_t)(rev->val >> 32), ids, n_ids, cap_ids);
        spm_resegment(c, (size_t)(uint32_t)(rev->val & 0xFFFFFFFF), ids, n_ids, cap_ids);
        return;
    }

    /* byte fallback */
    for (size_t j = 0; j < s->n; j++) {
        uint8_t byte = (uint8_t) s->text[j];
        int32_t bt = c->v->byte_tok[byte];
        if (bt >= 0) {
            si_push_id(ids, n_ids, cap_ids, (uint32_t) bt);
        } else if (!c->warned_no_byte_tok) {
            c->warned_no_byte_tok = true;
            si_log_warn("no byte-fallback token for byte 0x%02x; skipping it", byte);
        }
    }
}

void si_spm_encode(const si_vocab *v, const char *text, size_t len,
                   uint32_t **ids, size_t *n_ids, size_t *cap_ids) {
    spm_ctx c;
    memset(&c, 0, sizeof(c));
    c.v = v;

    /* split into symbols by UTF-8 lead-byte length (reference table) */
    c.n_syms = 0;
    size_t offs = 0;
    /* count first */
    while (offs < len) {
        size_t k = si_utf8_len_of((uint8_t) text[offs]);
        if (k > len - offs) k = len - offs;
        offs += k;
        c.n_syms++;
    }
    c.syms = si_xcalloc(c.n_syms ? c.n_syms : 1, sizeof(*c.syms));
    offs = 0;
    for (size_t i = 0; i < c.n_syms; i++) {
        size_t k = si_utf8_len_of((uint8_t) text[offs]);
        if (k > len - offs) k = len - offs;
        c.syms[i].text = text + offs;
        c.syms[i].n = k;
        c.syms[i].prev = (int32_t) i - 1;
        c.syms[i].next = (offs + k == len) ? -1 : (int32_t) i + 1;
        offs += k;
    }

    si_map_init(&c.rev_merge, c.n_syms + 16);

    for (size_t i = 1; i < c.n_syms; i++) {
        spm_try_add(&c, (int32_t) i - 1, (int32_t) i);
    }

    /* merge loop */
    {
        spm_bigram b;
        while (spm_heap_pop(&c.pq, &b)) {
            spm_sym *ls = &c.syms[b.left];
            spm_sym *rs = &c.syms[b.right];
            if (ls->n == 0 || rs->n == 0 || ls->n + rs->n != b.size) {
                continue;
            }
            ls->n += rs->n;
            rs->n = 0;
            ls->next = rs->next;
            if (rs->next >= 0) {
                c.syms[rs->next].prev = (int32_t) b.left;
            }
            spm_try_add(&c, ls->prev, (int32_t) b.left);
            spm_try_add(&c, (int32_t) b.left, ls->next);
        }
    }

    /* emit */
    for (int32_t i = 0; i != -1; i = c.syms[i].next) {
        if ((size_t) i >= c.n_syms) break;
        spm_resegment(&c, (size_t) i, ids, n_ids, cap_ids);
    }

    free(c.syms);
    free(c.pq.d);
    si_map_free(&c.rev_merge);
}
