/* siphon: tiny test framework (assert-counting, no dependencies) */
#ifndef SIPHON_TEST_UTIL_H
#define SIPHON_TEST_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int t_checks = 0;
static int t_fails = 0;

#define CHECK(cond) do { \
    t_checks++; \
    if (!(cond)) { \
        t_fails++; \
        fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

#define CHECK_STR(a, b) do { \
    t_checks++; \
    if ((a) == NULL || (b) == NULL || strcmp((a), (b)) != 0) { \
        t_fails++; \
        fprintf(stderr, "  FAIL %s:%d: \"%s\" != \"%s\"\n", __FILE__, __LINE__, (a) ? (a) : "(null)", (b) ? (b) : "(null)"); \
    } \
} while (0)

#define TEST_REPORT(name) do { \
    printf("%s: %d checks, %d failures\n", (name), t_checks, t_fails); \
    return t_fails ? 1 : 0; \
} while (0)

#endif
