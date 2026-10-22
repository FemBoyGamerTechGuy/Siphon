/* siphon: unicode classification + UTF-8 helpers for tokenizers.
 *
 * The tokenizer pretokenizers classify codepoints as letter/number/
 * whitespace and need byte-level BPE symbol mapping (GPT-2's
 * bytes-to-unicode). Tables are generated at development time
 * (scripts/gen_unicode_tables.py) and committed as C arrays.
 */
#ifndef SIPHON_UNICODE_H
#define SIPHON_UNICODE_H

#include "siphon/common.h"

/* Classification flags (values are part of the generated table format). */
#define SI_UNI_LETTER     0x1u
#define SI_UNI_NUMBER     0x2u
#define SI_UNI_WHITESPACE 0x4u
#define SI_UNI_DEFINED    0x8u

typedef struct {
    uint32_t first;   /* inclusive */
    uint32_t last;    /* inclusive */
    uint32_t flags;
} si_uni_range;

extern const si_uni_range si_uni_ranges[];
extern const uint32_t si_uni_ranges_n;

/* Classification for a codepoint (0 for unassigned codepoints). */
uint32_t si_uni_flags(uint32_t cp);
bool si_uni_is_letter(uint32_t cp);
bool si_uni_is_number(uint32_t cp);
bool si_uni_is_whitespace(uint32_t cp);
bool si_uni_is_defined(uint32_t cp);

/* UTF-8 ------------------------------------------------------------- */

/* Number of bytes of the UTF-8 sequence starting with `lead`, per the
 * reference's lead-nibble table. Continuation bytes and 0xF8+ map to
 * lengths 1 and 4 respectively (invalid input is handled by callers). */
size_t si_utf8_len_of(uint8_t lead);

/* Strict decode of one sequence. Returns bytes consumed, or 0 if the
 * sequence at s[0..len) is invalid (mirrors the reference decoder:
 * rejects overlongs, > 0x10FFFF, but accepts surrogates like it does). */
size_t si_utf8_decode(const uint8_t *s, size_t len, uint32_t *cp);

/* Encode cp into buf (needs up to 4 bytes). Returns bytes written. */
size_t si_utf8_encode(uint32_t cp, uint8_t *buf);

/* GPT-2 byte-level encoding ------------------------------------------ */

/* Byte -> unicode symbol (as UTF-8 bytes). Returns symbol length
 * (1 or 2). out must have room for 2 bytes + implicit NUL at out[2]. */
size_t si_uni_byte_to_sym(uint8_t byte, char out[3]);

/* Unicode symbol (as UTF-8 string) -> byte, or -1 if not a symbol that
 * maps back to a byte. */
int si_uni_sym_to_byte(const char *sym, size_t len);

#endif /* SIPHON_UNICODE_H */
