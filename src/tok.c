/* siphon: tokenize / detokenize entry points.
 *
 * Handles special-token partitioning of the input text, the per-family
 * pre/post processing (SPM space escaping and dummy prefix, BPE bos/eos
 * handling), and detokenization with the reference's piece rules and
 * whitespace cleanup passes.
 */
#include "tok_internal.h"

#include <ctype.h>

/* ---- special-token partition ----------------------------------------- */

typedef struct {
    bool is_token;
    int32_t token;
    size_t off, len;
} frag;

static const uint8_t *find_bytes(const uint8_t *hay, size_t haylen,
                                 const uint8_t *needle, size_t nlen) {
    if (nlen == 0 || haylen < nlen) return NULL;
    for (size_t i = 0; i + nlen <= haylen; i++) {
        if (hay[i] == needle[0] && memcmp(hay + i, needle, nlen) == 0) {
            return hay + i;
        }
    }
    return NULL;
}

/* Split raw fragments on special-token text. Specials are applied in
 * descending text-length order; control/unknown tokens only when
 * parse_special is set, user-defined always (reference behavior). */
static void partition(const si_vocab *v, const char *text, size_t len,
                      bool parse_special, frag **frags_out, size_t *n_frags_out,
                      size_t *cap_out) {
    frag *frags = si_xmalloc(8 * sizeof(*frags));
    size_t n = 0, cap = 8;
    frags[0].is_token = false;
    frags[0].token = -1;
    frags[0].off = 0;
    frags[0].len = len;
    n = 1;

    for (uint32_t s = 0; s < v->n_special; s++) {
        uint32_t sid = v->special_ids[s];
        uint8_t type = v->tok_type[sid];
        if (!parse_special && (type == SI_TOK_CONTROL || type == SI_TOK_UNKNOWN)) {
            continue;
        }
        const uint8_t *needle = (const uint8_t *) v->tok_text[sid];
        size_t nlen = v->tok_len[sid];
        if (nlen == 0) continue;

        size_t i = 0;
        while (i < n) {
            if (frags[i].is_token) { i++; continue; }
            const uint8_t *base = (const uint8_t *) text + frags[i].off;
            size_t blen = frags[i].len;
            const uint8_t *hit = find_bytes(base, blen, needle, nlen);
            if (!hit) { i++; continue; }

            size_t hit_off = (size_t)(hit - (const uint8_t *) text);
            size_t left_len = hit_off - frags[i].off;
            size_t right_off = hit_off + nlen;
            size_t right_len = frags[i].off + frags[i].len - right_off;

            /* replace frags[i] with [left?, token, right?] */
            size_t added = (left_len > 0) + 1 + (right_len > 0);
            if (n + added - 1 >= cap) {
                while (n + added - 1 >= cap) cap *= 2;
                frags = si_xrealloc(frags, cap * sizeof(*frags));
            }
            memmove(&frags[i + added], &frags[i + 1], (n - i - 1) * sizeof(*frags));

            size_t k = i;
            if (left_len > 0) {
                frags[k].is_token = false;
                frags[k].token = -1;
                frags[k].off = frags[i].off;
                frags[k].len = left_len;
                k++;
            }
            frags[k].is_token = true;
            frags[k].token = (int32_t) sid;
            frags[k].off = 0;
            frags[k].len = 0;
            k++;
            if (right_len > 0) {
                frags[k].is_token = false;
                frags[k].token = -1;
                frags[k].off = right_off;
                frags[k].len = right_len;
                k++;
            }
            n += added - 1;

            /* continue scanning at/after the token fragment */
            i = k;   /* == index after the token; right frag if present */
        }
    }

    *frags_out = frags;
    *n_frags_out = n;
    *cap_out = cap;
}

/* ---- tokenize --------------------------------------------------------- */

int si_tokenize(const si_vocab *v, const char *text, size_t len,
                bool add_special, bool parse_special,
                uint32_t **ids, size_t *n_ids) {
    uint32_t *out = NULL;
    size_t n_out = 0, cap = 0;
    (void) cap;

    frag *frags = NULL;
    size_t n_frags = 0, cap_frags = 0;
    if (len > 0) {
        partition(v, text, len, parse_special, &frags, &n_frags, &cap_frags);
    }

    if (v->kind == SI_VOCAB_SPM) {
        bool is_prev_special = true;
        if (add_special && v->add_bos && v->bos_id >= 0) {
            si_push_id(&out, &n_out, &cap, (uint32_t) v->bos_id);
            is_prev_special = true;
        }
        for (size_t i = 0; i < n_frags; i++) {
            if (!frags[i].is_token) {
                si_str s = {0};
                if (v->add_space_prefix && is_prev_special) {
                    si_str_push_byte(&s, ' ');
                }
                si_str_push(&s, text + frags[i].off, frags[i].len);
                /* escape spaces as U+2581 */
                si_str esc = {0};
                for (size_t j = 0; j < s.len; j++) {
                    if (s.d[j] == ' ') {
                        si_str_push(&esc, "\xe2\x96\x81", 3);
                    } else {
                        si_str_push_byte(&esc, s.d[j]);
                    }
                }
                free(s.d);
                si_spm_encode(v, esc.d, esc.len, &out, &n_out, &cap);
                free(esc.d);
                is_prev_special = false;
            } else {
                si_push_id(&out, &n_out, &cap, (uint32_t) frags[i].token);
                is_prev_special = true;
            }
        }
        if (add_special && v->add_eos && v->eos_id >= 0) {
            si_push_id(&out, &n_out, &cap, (uint32_t) v->eos_id);
        }
    } else {
        if (add_special && v->add_bos && v->bos_id >= 0) {
            si_push_id(&out, &n_out, &cap, (uint32_t) v->bos_id);
        }
        for (size_t i = 0; i < n_frags; i++) {
            if (!frags[i].is_token) {
                if (v->escape_ws) {
                    /* SPM-style BPE: escape first */
                    si_str esc = {0};
                    const char *f = text + frags[i].off;
                    for (size_t j = 0; j < frags[i].len; j++) {
                        if (f[j] == ' ') {
                            si_str_push(&esc, "\xe2\x96\x81", 3);
                        } else {
                            si_str_push_byte(&esc, f[j]);
                        }
                    }
                    si_bpe_encode(v, esc.d, esc.len, &out, &n_out, &cap);
                    free(esc.d);
                } else {
                    si_bpe_encode(v, text + frags[i].off, frags[i].len,
                                  &out, &n_out, &cap);
                }
            } else {
                si_push_id(&out, &n_out, &cap, (uint32_t) frags[i].token);
            }
        }
        if (add_special && v->add_eos && v->eos_id >= 0) {
            si_push_id(&out, &n_out, &cap, (uint32_t) v->eos_id);
        }
    }

    if (v->add_bos && v->bos_id >= 0 && n_out >= 2 && out[0] == (uint32_t) v->bos_id &&
        out[1] == (uint32_t) v->bos_id) {
        si_log_warn("prompt starts with a BOS token and another BOS was added");
    }

    free(frags);
    *ids = out;
    *n_ids = n_out;
    return SI_OK;
}

/* ---- detokenize ------------------------------------------------------- */

/* byte value of a BYTE-type token (SPM: <0xXX>; BPE: byte-level symbol) */
static int token_to_byte(const si_vocab *v, uint32_t id) {
    const char *t = v->tok_text[id];
    size_t tl = v->tok_len[id];
    if (v->kind == SI_VOCAB_SPM) {
        if (tl == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>' &&
            isxdigit((unsigned char) t[3]) && isxdigit((unsigned char) t[4])) {
            int hi = isdigit((unsigned char) t[3]) ? t[3] - '0' : (tolower(t[3]) - 'a' + 10);
            int lo = isdigit((unsigned char) t[4]) ? t[4] - '0' : (tolower(t[4]) - 'a' + 10);
            return hi * 16 + lo;
        }
        if (tl == 1) return (uint8_t) t[0];
        return -1;
    }
    return si_uni_sym_to_byte(t, tl);
}

static void push_lstripped(si_str *out, const char *p, size_t n, int lstrip) {
    for (int i = 0; i < lstrip && n > 0 && p[0] == ' '; i++) {
        p++;
        n--;
    }
    si_str_push(out, p, n);
}

static void push_piece(const si_vocab *v, uint32_t id, int lstrip, bool special,
                       si_str *out) {
    if (id >= v->n_tokens) return;
    uint8_t type = v->tok_type[id];
    const char *t = v->tok_text[id];
    size_t tl = v->tok_len[id];

    if (!special && (type == SI_TOK_UNKNOWN || type == SI_TOK_CONTROL)) {
        return;  /* suppressed */
    }
    if (type == SI_TOK_CONTROL || type == SI_TOK_UNKNOWN || type == SI_TOK_USER) {
        push_lstripped(out, t, tl, lstrip);
        return;
    }
    if (type == SI_TOK_NORMAL) {
        if (v->kind == SI_VOCAB_SPM || v->escape_ws) {
            /* replace U+2581 with space, then lstrip */
            si_str r = {0};
            size_t i = 0;
            while (i < tl) {
                if (i + 3 <= tl && (uint8_t) t[i] == 0xE2 && (uint8_t) t[i+1] == 0x96 &&
                    (uint8_t) t[i+2] == 0x81) {
                    si_str_push_byte(&r, ' ');
                    i += 3;
                } else {
                    si_str_push_byte(&r, t[i]);
                    i++;
                }
            }
            push_lstripped(out, r.d, r.len, lstrip);
            free(r.d);
            return;
        }
        /* byte-level decode */
        size_t off = 0;
        while (off < tl) {
            uint32_t cp;
            size_t k = si_utf8_decode((const uint8_t *) t + off, tl - off, &cp);
            if (k == 0) {
                cp = 0xFFFD;  /* matches the reference's lenient decode */
                k = 1;
            }
            char sym[5];
            size_t sl = si_utf8_encode(cp, (uint8_t *) sym);
            int byte = si_uni_sym_to_byte(sym, sl);
            if (byte >= 0) {
                si_str_push_byte(out, (char) byte);
            } else {
                /* reference renders unknown symbols verbosely */
                static const char hex[] = "0123456789abcdef";
                si_str_push(out, "[UNK_BYTE_0x", 12);
                for (size_t j = 0; j < sl; j++) {
                    char h[2] = { hex[(uint8_t) sym[j] >> 4], hex[(uint8_t) sym[j] & 15] };
                    si_str_push(out, h, 2);
                }
                si_str_push(out, t, tl);
                si_str_push(out, "]", 1);
            }
            off += k;
        }
        return;
    }
    if (type == SI_TOK_BYTE) {
        int byte = token_to_byte(v, id);
        if (byte >= 0) {
            char c = (char) byte;
            push_lstripped(out, &c, 1, lstrip);
        }
        return;
    }
    /* UNUSED / undefined types: empty piece, like the reference */
}

/* The reference's "clean spaces" detokenize cleanup (used by gpt2-style
 * BPE models): drop spaces before ?!., and reshape apostrophe spacing. */
static void clean_spaces_passes(si_str *s) {
    if (s->len == 0) return;

    /* pass 1: remove space before ? ! . , */
    {
        size_t w = 1;
        for (size_t i = 1; i < s->len; i++) {
            char x = s->d[i];
            if (s->d[i - 1] == ' ' &&
                (x == '?' || x == '!' || x == '.' || x == ',')) {
                w--;  /* drop the space */
            }
            s->d[w++] = x;
        }
        s->len = w;
    }
    /* pass 2: " ' " -> "'" */
    {
        size_t w = 1;
        for (size_t i = 1; i < s->len; i++) {
            char x = s->d[i];
            if (x == '\'' && i + 1 < s->len && s->d[i - 1] == ' ' && s->d[i + 1] == ' ') {
                w--;
                i++;  /* skip the trailing space */
            }
            s->d[w++] = x;
        }
        s->len = w;
    }
    /* pass 3: apostrophe contractions (" 's"/" 'm"/" 're"/" 've" lose
     * their space; " 't"/" 'd"/" 'll" keep it) */
    {
        size_t w = 1;
        for (size_t i = 1; i < s->len; i++) {
            char x = s->d[i];
            bool drop = false;
            if (s->d[i - 1] == ' ' && x == '\'' && i + 1 < s->len) {
                char x1 = s->d[i + 1];
                if (x1 == 't' || x1 == 'd') {
                    drop = false;
                } else if (x1 == 's' || x1 == 'm') {
                    drop = true;
                } else if (i + 2 < s->len) {
                    char x2 = s->d[i + 2];
                    if (x1 == 'l' && x2 == 'l') {
                        drop = false;
                    } else if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {
                        drop = true;
                    }
                }
            }
            if (drop) w--;
            s->d[w++] = x;
        }
        s->len = w;
    }
}

int si_detokenize(const si_vocab *v, const uint32_t *ids, size_t n_ids,
                  bool remove_special, bool unparse_special,
                  char **text, size_t *len) {
    const uint32_t *ids_in = ids;
    size_t n = n_ids;

    bool remove_space = v->add_space_prefix;
    if (remove_special && v->add_bos && v->bos_id >= 0 && n > 0 &&
        ids[0] == (uint32_t) v->bos_id) {
        remove_space = false;
        ids++;
        n--;
    }
    if (remove_special && v->add_eos && v->eos_id >= 0 && n > 0 &&
        ids[n - 1] == (uint32_t) v->eos_id) {
        n--;
    }

    si_str s = {0};
    for (size_t i = 0; i < n; i++) {
        int lstrip = (i == 0 && remove_space) ? 1 : 0;
        push_piece(v, ids[i], lstrip, unparse_special, &s);
    }

    if (v->clean_spaces) {
        clean_spaces_passes(&s);
    }

    si_str_push_byte(&s, '\0');
    if (s.d == NULL) {
        /* no tokens at all: empty string */
        s.d = si_xstrdup("");
        s.len = 1;
    }
    *text = s.d;
    *len = s.len - 1;
    (void) ids_in;
    (void) n_ids;
    return SI_OK;
}
