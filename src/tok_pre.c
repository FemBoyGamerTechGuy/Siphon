/* siphon: BPE pretokenizer — regex-equivalent matchers.
 *
 * Hand-written matchers that mirror the reference tokenizer's custom
 * implementations exactly (including their whitespace-run and
 * invalid-UTF-8 behavior), for the pretokenizer flavors real models
 * use: gpt-2, qwen2 (Qwen2/2.5/3), llama3, and a default fallback.
 *
 * Pipeline (same as reference):
 *   text --decode--> codepoints (invalid bytes -> U+FFFD, one per byte)
 *        --match--> word boundaries in codepoint space
 *        --encode--> utf8 words, then GPT-2 byte-level encoding per word
 */
#include "siphon/unicode.h"

#include "tok_internal.h"

/* decode text to codepoints; invalid sequences advance one byte and
 * emit U+FFFD (reference behavior) */
static size_t decode_cpts(const char *text, size_t len, uint32_t **out) {
    uint32_t *cpts = si_xmalloc((len + 1) * sizeof(uint32_t));
    uint32_t *fl_unused = NULL;
    (void) fl_unused;
    size_t n = 0, off = 0;
    while (off < len) {
        uint32_t cp;
        size_t k = si_utf8_decode((const uint8_t *) text + off, len - off, &cp);
        if (k == 0) {
            cpts[n++] = 0xFFFD;
            off += 1;
        } else {
            cpts[n++] = cp;
            off += k;
        }
    }
    *out = cpts;
    return n;
}

/* per-codepoint flags cache; entry n is 0 (out of range => no flags) */
static uint32_t *flags_cache(const uint32_t *cpts, size_t n) {
    uint32_t *fl = si_xmalloc((n + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < n; i++) fl[i] = si_uni_flags(cpts[i]);
    fl[n] = 0;
    return fl;
}

/* growable word storage */
static void words_push(si_words *w, const uint32_t *cpts, size_t a, size_t b,
                       bool byte_encode) {
    /* build the utf8 word for cpts[a..b) */
    char tmp[8];
    size_t start = w->buf.len;
    for (size_t i = a; i < b; i++) {
        size_t k = si_utf8_encode(cpts[i], (uint8_t *) tmp);
        if (w->buf.len + k > w->buf.cap) {
            w->buf.cap = (w->buf.len + k) * 2 + 64;
            w->buf.d = si_xrealloc(w->buf.d, w->buf.cap);
        }
        memcpy(w->buf.d + w->buf.len, tmp, k);
        w->buf.len += k;
    }
    size_t utf8_len = w->buf.len - start;

    if (!byte_encode) {
        w->lens[w->n] = utf8_len;
        w->offs[w->n] = start;
        w->n++;
        return;
    }

    /* byte-level encode in place into a scratch region (each byte grows
     * to at most 2 bytes) */
    size_t enc_cap = utf8_len * 2;
    char *enc = si_xmalloc(enc_cap ? enc_cap : 1);
    size_t enc_len = 0;
    for (size_t i = start; i < start + utf8_len; i++) {
        char sym[3];
        size_t sl = si_uni_byte_to_sym((uint8_t) w->buf.d[i], sym);
        memcpy(enc + enc_len, sym, sl);
        enc_len += sl;
    }
    /* replace the raw word with the encoded one */
    w->buf.len = start;
    if (w->buf.len + enc_len > w->buf.cap) {
        w->buf.cap = (w->buf.len + enc_len) * 2 + 64;
        w->buf.d = si_xrealloc(w->buf.d, w->buf.cap);
    }
    memcpy(w->buf.d + start, enc, enc_len);
    w->buf.len += enc_len;
    free(enc);

    w->offs[w->n] = start;
    w->lens[w->n] = enc_len;
    w->n++;
}

static void words_reserve(si_words *w, size_t cap) {
    w->offs = si_xrealloc(w->offs, cap * sizeof(size_t));
    w->lens = si_xrealloc(w->lens, cap * sizeof(size_t));
}

/* ---- gpt2 matcher ----------------------------------------------------
 * regex: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 * (case-sensitive contractions) */
static void split_gpt2(const uint32_t *cpts, const uint32_t *fl, size_t n,
                       bool byte_encode, si_words *w) {
    size_t prev_end = 0;
    size_t pos = 0;
    while (pos < n) {
        uint32_t cpt = cpts[pos];
        uint32_t flags = fl[pos];

        /* contractions */
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t nxt = cpts[pos + 1];
            if (nxt == 's' || nxt == 't' || nxt == 'm' || nxt == 'd') {
                words_push(w, cpts, prev_end, pos + 2, byte_encode);
                prev_end = pos = pos + 2;
                continue;
            }
            if (pos + 2 < n) {
                uint32_t nxnx = cpts[pos + 2];
                if ((nxt == 'r' && nxnx == 'e') || (nxt == 'v' && nxnx == 'e') ||
                    (nxt == 'l' && nxnx == 'l')) {
                    words_push(w, cpts, prev_end, pos + 3, byte_encode);
                    prev_end = pos = pos + 3;
                    continue;
                }
            }
        }

        uint32_t flags2 = (cpt == ' ') ? fl[pos + 1] : flags;
        /* ?\p{L}+ */
        if (flags2 & SI_UNI_LETTER) {
            pos += (cpt == ' ');
            while (fl[pos] & SI_UNI_LETTER) pos++;
            words_push(w, cpts, prev_end, pos, byte_encode);
            prev_end = pos;
            continue;
        }
        /* ?\p{N}+ */
        if (flags2 & SI_UNI_NUMBER) {
            pos += (cpt == ' ');
            while (fl[pos] & SI_UNI_NUMBER) pos++;
            words_push(w, cpts, prev_end, pos, byte_encode);
            prev_end = pos;
            continue;
        }
        /* ?[^\s\p{L}\p{N}]+ (defined codepoints only) */
        if (!(flags2 & (SI_UNI_WHITESPACE | SI_UNI_LETTER | SI_UNI_NUMBER)) &&
            flags2 != 0) {
            pos += (cpt == ' ');
            while (!(fl[pos] & (SI_UNI_WHITESPACE | SI_UNI_LETTER | SI_UNI_NUMBER)) &&
                   fl[pos] != 0) {
                pos++;
            }
            words_push(w, cpts, prev_end, pos, byte_encode);
            prev_end = pos;
            continue;
        }

        /* whitespace */
        size_t num_ws = 0;
        while (fl[pos + num_ws] & SI_UNI_WHITESPACE) num_ws++;
        /* \s+(?!\S): leave the last ws to join the next word (when one exists) */
        if (num_ws > 1 && pos + num_ws < n) {
            pos += num_ws - 1;
            words_push(w, cpts, prev_end, pos, byte_encode);
            prev_end = pos;
            continue;
        }
        /* \s+ */
        if (num_ws > 0) {
            pos += num_ws;
            words_push(w, cpts, prev_end, pos, byte_encode);
            prev_end = pos;
            continue;
        }

        /* no match: single codepoint */
        pos++;
        words_push(w, cpts, prev_end, pos, byte_encode);
        prev_end = pos;
    }
}

/* ---- qwen2 / llama3 matcher ------------------------------------------ */
static void split_qwen2_like(const uint32_t *cpts, const uint32_t *fl, size_t n,
                             bool digits_max3, bool byte_encode, si_words *w) {
    size_t prev_end = 0;
    size_t pos = 0;
    while (pos < n) {
        uint32_t cpt = cpts[pos];
        uint32_t flags = fl[pos];

        /* (?i:'s|'t|'re|'ve|'m|'ll|'d) */
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t nxt = (uint32_t) tolower((int) cpts[pos + 1]);
            if (nxt == 's' || nxt == 't' || nxt == 'm' || nxt == 'd') {
                words_push(w, cpts, prev_end, pos + 2, byte_encode);
                prev_end = pos = pos + 2;
                continue;
            }
            if (pos + 2 < n) {
                uint32_t nxnx = (uint32_t) tolower((int) cpts[pos + 2]);
                if ((nxt == 'r' && nxnx == 'e') || (nxt == 'v' && nxnx == 'e') ||
                    (nxt == 'l' && nxnx == 'l')) {
                    words_push(w, cpts, prev_end, pos + 3, byte_encode);
                    prev_end = pos = pos + 3;
                    continue;
                }
            }
        }

        /* [^\r\n\p{L}\p{N}]?\p{L}+ */
        if (!(cpt == '\r' || cpt == '\n' || (flags & SI_UNI_NUMBER))) {
            if ((flags & SI_UNI_LETTER) || (fl[pos + 1] & SI_UNI_LETTER)) {
                pos++;
                while (fl[pos] & SI_UNI_LETTER) pos++;
                words_push(w, cpts, prev_end, pos, byte_encode);
                prev_end = pos;
                continue;
            }
        }

        /* \p{N} or \p{N}{1,3} */
        if (flags & SI_UNI_NUMBER) {
            if (digits_max3) {
                size_t ini = pos;
                while (fl[pos] & SI_UNI_NUMBER) {
                    pos++;
                    if (pos - ini >= 3) {
                        words_push(w, cpts, prev_end, pos, byte_encode);
                        prev_end = ini = pos;
                    }
                }
                words_push(w, cpts, prev_end, pos, byte_encode);
                prev_end = pos;
            } else {
                pos++;
                words_push(w, cpts, prev_end, pos, byte_encode);
                prev_end = pos;
            }
            continue;
        }

        /* ?[^\s\p{L}\p{N}]+[\r\n]* */
        {
            uint32_t flags2 = (cpt == ' ') ? fl[pos + 1] : flags;
            if (!(flags2 & (SI_UNI_WHITESPACE | SI_UNI_LETTER | SI_UNI_NUMBER)) &&
                flags != 0) {
                pos += (cpt == ' ');
                while (!(fl[pos] & (SI_UNI_WHITESPACE | SI_UNI_LETTER | SI_UNI_NUMBER)) &&
                       fl[pos] != 0) {
                    pos++;
                }
                while (pos < n && (cpts[pos] == '\r' || cpts[pos] == '\n')) pos++;
                words_push(w, cpts, prev_end, pos, byte_encode);
                prev_end = pos;
                continue;
            }
        }

        /* whitespace */
        size_t num_ws = 0;
        size_t last_end_rn = 0;
        while (fl[pos + num_ws] & SI_UNI_WHITESPACE) {
            uint32_t cpt2 = cpts[pos + num_ws];
            if (cpt2 == '\r' || cpt2 == '\n') {
                last_end_rn = pos + num_ws + 1;
            }
            num_ws++;
        }
        /* \s*[\r\n]+ */
        if (last_end_rn > 0) {
            pos = last_end_rn;
            words_push(w, cpts, prev_end, pos, byte_encode);
            prev_end = pos;
            continue;
        }
        /* \s+(?!\S) */
        if (num_ws > 1 && pos + num_ws < n) {
            pos += num_ws - 1;
            words_push(w, cpts, prev_end, pos, byte_encode);
            prev_end = pos;
            continue;
        }
        /* \s+ */
        if (num_ws > 0) {
            pos += num_ws;
            words_push(w, cpts, prev_end, pos, byte_encode);
            prev_end = pos;
            continue;
        }

        /* no match */
        pos++;
        words_push(w, cpts, prev_end, pos, byte_encode);
        prev_end = pos;
    }
}

int si_pre_split(const char *text, size_t len, si_pre_kind pre, bool byte_encode,
                 si_words *w) {
    memset(w, 0, sizeof(*w));
    if (len == 0) return SI_OK;

    uint32_t *cpts = NULL;
    size_t n = decode_cpts(text, len, &cpts);
    uint32_t *fl = flags_cache(cpts, n);

    /* worst case: one word per codepoint, plus slack */
    words_reserve(w, n + 8);

    switch (pre) {
        case SI_PRE_GPT2:
            split_gpt2(cpts, fl, n, byte_encode, w);
            break;
        case SI_PRE_QWEN2:
            split_qwen2_like(cpts, fl, n, false, byte_encode, w);
            break;
        case SI_PRE_LLAMA3:
            split_qwen2_like(cpts, fl, n, true, byte_encode, w);
            break;
        case SI_PRE_DEFAULT:
        default:
            /* reference 'default' chains extra number/punct refinement
             * regexes; we use the gpt2 matcher as the close fallback */
            split_gpt2(cpts, fl, n, byte_encode, w);
            break;
    }

    free(cpts);
    free(fl);
    return SI_OK;
}

void si_words_free(si_words *w) {
    free(w->buf.d);
    free(w->offs);
    free(w->lens);
    memset(w, 0, sizeof(*w));
}
