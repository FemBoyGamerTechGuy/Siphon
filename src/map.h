/* siphon: internal byte-string -> uint64 hash map (open addressing).
 * Private to the tokenizer implementation. */
#ifndef SIPHON_MAP_H
#define SIPHON_MAP_H

#include "siphon/common.h"

typedef struct {
    char *k;              /* owned copy of the key */
    uint32_t klen;
    bool used;
    uint64_t val;
} si_map_slot;

typedef struct {
    si_map_slot *slots;
    uint32_t cap;        /* power of two */
    uint32_t count;
} si_map;

static inline uint64_t si_map_hash(const char *k, uint32_t klen) {
    /* FNV-1a 64 */
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < klen; i++) {
        h ^= (uint8_t) k[i];
        h *= 1099511628211ull;
    }
    return h;
}

static inline void si_map_init(si_map *m, uint32_t expect) {
    uint32_t cap = 16;
    while (cap < expect * 2) cap <<= 1;
    m->slots = si_xcalloc(cap, sizeof(*m->slots));
    m->cap = cap;
    m->count = 0;
}

static inline void si_map_free(si_map *m) {
    for (uint32_t i = 0; i < m->cap; i++) {
        free(m->slots[i].k);
    }
    free(m->slots);
    m->slots = NULL;
    m->cap = 0;
    m->count = 0;
}

/* Get; returns NULL if absent. */
static inline si_map_slot *si_map_get(si_map *m, const char *k, uint32_t klen) {
    if (m->cap == 0) return NULL;
    uint64_t h = si_map_hash(k, klen);
    uint32_t i = (uint32_t) h & (m->cap - 1);
    for (;;) {
        si_map_slot *s = &m->slots[i];
        if (!s->used) return NULL;
        if (s->klen == klen && memcmp(s->k, k, klen) == 0) return s;
        i = (i + 1) & (m->cap - 1);
    }
}

/* Insert or update: overwrite=true replaces an existing value
 * (mirrors map[k]=v); overwrite=false keeps the existing value
 * (mirrors emplace). Keys are copied; the map owns them. */
static inline void si_map_put(si_map *m, const char *k, uint32_t klen,
                              uint64_t val, bool overwrite) {
    if (m->count * 2 >= m->cap) {
        /* grow: rehash (keys are re-owned by the new table) */
        si_map nm;
        si_map_init(&nm, m->cap ? m->cap : 16);
        for (uint32_t i = 0; i < m->cap; i++) {
            if (m->slots[i].used) {
                si_map_put(&nm, m->slots[i].k, m->slots[i].klen,
                           m->slots[i].val, true);
            }
        }
        for (uint32_t i = 0; i < m->cap; i++) {
            free(m->slots[i].k);
        }
        free(m->slots);
        *m = nm;
    }
    uint64_t h = si_map_hash(k, klen);
    uint32_t i = (uint32_t) h & (m->cap - 1);
    for (;;) {
        si_map_slot *s = &m->slots[i];
        if (!s->used) {
            s->used = true;
            s->k = si_xmalloc(klen);
            memcpy(s->k, k, klen);
            s->klen = klen;
            s->val = val;
            m->count++;
            return;
        }
        if (s->klen == klen && memcmp(s->k, k, klen) == 0) {
            if (overwrite) s->val = val;
            return;
        }
        i = (i + 1) & (m->cap - 1);
    }
}

#endif /* SIPHON_MAP_H */
