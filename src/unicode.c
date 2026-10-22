/* siphon: unicode classification + UTF-8 helpers for tokenizers */
#include "siphon/unicode.h"

/* ---- classification (binary search over generated ranges) ---------- */

uint32_t si_uni_flags(uint32_t cp) {
    /* binary search: ranges are sorted and non-overlapping */
    uint32_t lo = 0, hi = si_uni_ranges_n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        const si_uni_range *r = &si_uni_ranges[mid];
        if (cp < r->first) {
            hi = mid;
        } else if (cp > r->last) {
            lo = mid + 1;
        } else {
            return r->flags;
        }
    }
    return 0;
}

bool si_uni_is_letter(uint32_t cp)     { return (si_uni_flags(cp) & SI_UNI_LETTER) != 0; }
bool si_uni_is_number(uint32_t cp)     { return (si_uni_flags(cp) & SI_UNI_NUMBER) != 0; }
bool si_uni_is_whitespace(uint32_t cp) { return (si_uni_flags(cp) & SI_UNI_WHITESPACE) != 0; }
bool si_uni_is_defined(uint32_t cp)    { return (si_uni_flags(cp) & SI_UNI_DEFINED) != 0; }

/* ---- UTF-8 ---------------------------------------------------------- */

size_t si_utf8_len_of(uint8_t lead) {
    /* same lookup table the reference uses: indexed by high nibble */
    static const size_t lookup[16] = { 1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4 };
    return lookup[lead >> 4];
}

size_t si_utf8_decode(const uint8_t *s, size_t len, uint32_t *cp) {
    if (len == 0) return 0;
    uint8_t c0 = s[0];
    if (!(c0 & 0x80)) {
        *cp = c0;
        return 1;
    }
    if (!(c0 & 0x40)) return 0;             /* stray continuation */
    if (!(c0 & 0x20)) {
        if (len < 2 || (s[1] & 0xC0) != 0x80) return 0;
        uint32_t v = ((uint32_t)(c0 & 0x1F) << 6) | (s[1] & 0x3F);
        if (v < 0x80) return 0;             /* overlong */
        *cp = v;
        return 2;
    }
    if (!(c0 & 0x10)) {
        if (len < 3 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return 0;
        uint32_t v = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (v < 0x800) return 0;            /* overlong */
        *cp = v;
        return 3;
    }
    if (!(c0 & 0x08)) {
        if (len < 4 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 ||
            (s[3] & 0xC0) != 0x80) return 0;
        uint32_t v = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
                     ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (v < 0x10000 || v > 0x10FFFF) return 0;
        *cp = v;
        return 4;
    }
    return 0;                               /* 5- or 6-byte forms are invalid */
}

size_t si_utf8_encode(uint32_t cp, uint8_t *buf) {
    if (cp < 0x80) {
        buf[0] = (uint8_t) cp;
        return 1;
    }
    if (cp < 0x800) {
        buf[0] = (uint8_t) (0xC0 | (cp >> 6));
        buf[1] = (uint8_t) (0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        buf[0] = (uint8_t) (0xE0 | (cp >> 12));
        buf[1] = (uint8_t) (0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (uint8_t) (0x80 | (cp & 0x3F));
        return 3;
    }
    buf[0] = (uint8_t) (0xF0 | (cp >> 18));
    buf[1] = (uint8_t) (0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (uint8_t) (0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (uint8_t) (0x80 | (cp & 0x3F));
    return 4;
}

/* ---- GPT-2 byte-level encoding -------------------------------------- */

/* Which bytes map to themselves in the byte-level alphabet:
 * 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF. The others map to U+0100 + n
 * where n counts the unmapped bytes in order. */
static bool byte_maps_to_self(uint8_t b) {
    return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
}

size_t si_uni_byte_to_sym(uint8_t byte, char out[3]) {
    if (byte_maps_to_self(byte)) {
        /* the symbol is the codepoint `byte` itself: 1 byte for ASCII,
         * 2 bytes (0xC2/0xC3 lead) for the 0xA1..0xFF ranges */
        size_t k = si_utf8_encode(byte, (uint8_t *) out);
        out[k] = '\0';
        return k;
    }
    /* count how many bytes before `byte` also needed remapping */
    uint32_t n = 0;
    for (int b = 0; b < (int) byte; b++) {
        if (!byte_maps_to_self((uint8_t) b)) n++;
    }
    uint32_t cp = 0x100 + n;
    size_t k = si_utf8_encode(cp, (uint8_t *) out);
    out[k] = '\0';
    return k;
}

int si_uni_sym_to_byte(const char *sym, size_t len) {
    /* symbols are the UTF-8 encoding of a codepoint: either a
     * self-mapped byte value (U+0021..U+007E, U+00A1..U+00AC,
     * U+00AE..U+00FF) or U+0100..U+0143 for the remapped bytes */
    uint32_t cp;
    size_t n = si_utf8_decode((const uint8_t *) sym, len, &cp);
    if (n != len || n == 0 || n > 2) return -1;
    if (cp < 0x100) {
        return byte_maps_to_self((uint8_t) cp) ? (int) cp : -1;
    }
    if (cp < 0x100 + 68) {  /* 68 remapped bytes total */
        uint32_t want = cp - 0x100;
        uint32_t n_seen = 0;
        for (int b = 0; b < 256; b++) {
            if (!byte_maps_to_self((uint8_t) b)) {
                if (n_seen == want) return b;
                n_seen++;
            }
        }
    }
    return -1;
}
