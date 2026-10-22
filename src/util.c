/* siphon: util — error strings, logging, allocation, misc helpers */
#include "siphon/common.h"

#include <stdarg.h>
#include <time.h>

static char g_errstr[512];

const char *si_errstr(void) {
    return g_errstr;
}

void si_set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_errstr, sizeof(g_errstr), fmt, ap);
    va_end(ap);
}

void si_log_warn(const char *fmt, ...) {
    va_list ap;
    fputs("siphon: warning: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

void si_log_err(const char *fmt, ...) {
    va_list ap;
    fputs("siphon: error: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

void *si_xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p && n) {
        fprintf(stderr, "siphon: fatal: out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}

void *si_xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p && n && sz) {
        fprintf(stderr, "siphon: fatal: out of memory (%zu x %zu bytes)\n", n, sz);
        abort();
    }
    return p;
}

void *si_xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q && n) {
        fprintf(stderr, "siphon: fatal: out of memory (%zu bytes)\n", n);
        abort();
    }
    return q;
}

char *si_xstrdup(const char *s) {
    char *p = strdup(s);
    if (!p) {
        fprintf(stderr, "siphon: fatal: out of memory (strdup)\n");
        abort();
    }
    return p;
}

char *si_xstrndup(const char *s, size_t n) {
    char *p = si_xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

char *si_fmt_size(char *buf, size_t buflen, uint64_t bytes) {
    const char *unit = "B";
    double val = (double) bytes;
    if (val >= 1024.0 * 1024.0 * 1024.0 * 1024.0) {
        val /= 1024.0 * 1024.0 * 1024.0 * 1024.0;
        unit = "TiB";
    } else if (val >= 1024.0 * 1024.0 * 1024.0) {
        val /= 1024.0 * 1024.0 * 1024.0;
        unit = "GiB";
    } else if (val >= 1024.0 * 1024.0) {
        val /= 1024.0 * 1024.0;
        unit = "MiB";
    } else if (val >= 1024.0) {
        val /= 1024.0;
        unit = "KiB";
    }
    snprintf(buf, buflen, "%.2f %s", val, unit);
    return buf;
}

double si_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}
