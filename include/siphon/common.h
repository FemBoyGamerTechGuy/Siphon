/* siphon: common definitions
 *
 * Siphon is a from-scratch GGUF inference engine in raw C for Linux.
 * This header provides the small shared foundation: error reporting,
 * logging, and allocation helpers. Everything here is dependency-free
 * (libc + libm + pthreads only, per the project rules).
 */
#ifndef SIPHON_COMMON_H
#define SIPHON_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Version of the siphon binary. Bump when features land. */
#define SIPHON_VERSION "0.1.0"

/* Error codes. Functions return 0 on success, negative on failure.
 * A human-readable message for the most recent failure is kept in
 * si_errstr() (not thread-safe; milestone 5 introduces per-thread
 * state where needed). */
enum {
    SI_OK            = 0,
    SI_ENOMEM       = -1,
    SI_EIO          = -2,
    SI_EFORMAT      = -3,  /* file format / parsing error */
    SI_ENOTSUP      = -4,  /* recognized but unsupported */
    SI_EINVAL       = -5,
    SI_ENOENT       = -6,
    SI_ERANGE       = -7,
};

/* Last error message (static buffer, valid until next si_set_err). */
const char *si_errstr(void);
void si_set_err(const char *fmt, ...);

/* Logging to stderr, always prefixed so users can grep for it. */
void si_log_warn(const char *fmt, ...);
void si_log_err(const char *fmt, ...);

/* Allocation helpers that abort on OOM. For a local inference engine
 * an OOM is unrecoverable anyway; dying with a clear message beats
 * returning NULL and having every caller handle it. */
void *si_xmalloc(size_t n);
void *si_xcalloc(size_t n, size_t sz);
void *si_xrealloc(void *p, size_t n);
char *si_xstrdup(const char *s);
char *si_xstrndup(const char *s, size_t n);

/* Human-readable byte counts: "637.30 MiB". Writes into buf (>= 32 bytes). */
char *si_fmt_size(char *buf, size_t buflen, uint64_t bytes);

/* Monotonic-ish wall clock for honest benchmark reporting. */
double si_time_sec(void);

#endif /* SIPHON_COMMON_H */
