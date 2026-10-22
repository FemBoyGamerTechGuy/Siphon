/* siphon: threadpool tests
 *
 * Proves the invariance contract: a task that partitions work strictly
 * by (lane, n_lanes) produces identical results for every lane count,
 * dispatches are serialized correctly, single-lane pools run inline,
 * and rapid consecutive dispatches stay consistent.
 */
#include "test_util.h"
#include "siphon/threadpool.h"

#include <math.h>

static uint64_t rng_state = 0x1234567;
static uint64_t trand(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}

/* deterministic integer dot of a strided slice, executed per lane.
 * Integer accumulation is exact regardless of partition, which is the
 * same contract the engine relies on: one output row fully computed
 * inside one lane. */
typedef struct {
    const int64_t *a;
    const int64_t *b;
    int n;
    int64_t *slots;   /* n_lanes results, summed at the end */
} dot_task;

static void dot_task_fn(void *ctxp, int lane, int lanes) {
    dot_task *t = ctxp;
    int64_t s = 0;
    for (int i = lane; i < t->n; i += lanes) {
        s += t->a[i] * t->b[i];
    }
    t->slots[lane] = s;
}

static int64_t run_dot(si_tp *tp, const int64_t *a, const int64_t *b, int n,
                       int64_t *slots) {
    dot_task t = { .a = a, .b = b, .n = n, .slots = slots };
    si_tp_run(tp, dot_task_fn, &t);
    int64_t s = 0;
    for (int i = 0; i < si_tp_nlanes(tp); i++) s += slots[i];
    return s;
}

static void test_lane_invariance(void) {
    enum { N = 977 };
    static int64_t a[N], b[N];
    for (int i = 0; i < N; i++) {
        a[i] = (int64_t)(trand() >> 33) - 4000000;
        b[i] = (int64_t)(trand() >> 33) - 4000000;
    }

    int64_t ref = 0;
    for (int i = 0; i < N; i++) ref += a[i] * b[i];

    int64_t slots[8];
    for (int lanes = 1; lanes <= 4; lanes++) {
        si_tp *tp = si_tp_create(lanes);
        CHECK(si_tp_nlanes(tp) == lanes);
        int64_t got = run_dot(tp, a, b, N, slots);
        CHECK(got == ref);   /* exact for any lane count */
        si_tp_free(tp);
    }
}

static void test_rapid_dispatch(void) {
    enum { N = 64, RUNS = 400 };
    static int64_t a[N], b[N];
    for (int i = 0; i < N; i++) {
        a[i] = (int64_t)(i + 1);
        b[i] = (int64_t)(1000 - i);
    }
    int64_t ref = 0;
    for (int i = 0; i < N; i++) ref += a[i] * b[i];

    si_tp *tp = si_tp_create(2);
    int64_t slots[4];
    for (int r = 0; r < RUNS; r++) {
        CHECK(run_dot(tp, a, b, N, slots) == ref);
    }
    si_tp_free(tp);
}

typedef struct {
    int8_t seen[8];
    int n_lanes;
} lane_task;

static void lane_task_fn(void *ctxp, int lane, int lanes) {
    lane_task *t = ctxp;
    (void)lanes;
    t->seen[lane] = 1;
}

static void test_all_lanes_run(void) {
    for (int lanes = 1; lanes <= 4; lanes++) {
        lane_task t = {0};
        t.n_lanes = lanes;
        si_tp *tp = si_tp_create(lanes);
        si_tp_run(tp, lane_task_fn, &t);
        for (int i = 0; i < lanes; i++) CHECK(t.seen[i] == 1);
        for (int i = lanes; i < 8; i++) CHECK(t.seen[i] == 0);
        si_tp_free(tp);
    }
}

/* per-lane scratch isolation: each lane writes its own slot only */
typedef struct {
    double *slots;
    int64_t stamp;
} iso_task;

static void iso_task_fn(void *ctxp, int lane, int lanes) {
    iso_task *t = ctxp;
    (void)lanes;
    t->slots[lane] = (double)t->stamp;
}

static void test_scratch_isolation(void) {
    for (int lanes = 2; lanes <= 4; lanes++) {
        double slots[4] = {0, 0, 0, 0};
        si_tp *tp = si_tp_create(lanes);
        for (int r = 0; r < 100; r++) {
            iso_task t = { .slots = slots, .stamp = 0x1000 + r };
            si_tp_run(tp, iso_task_fn, &t);
            for (int i = 0; i < lanes; i++) {
                CHECK(slots[i] == (double)(0x1000 + r));
            }
        }
        si_tp_free(tp);
    }
}

/* heavy stress with an eye on the spin-then-park path */
typedef struct {
    float *buf;
    int n;
} burn_task;

static void burn_task_fn(void *ctxp, int lane, int lanes) {
    burn_task *t = ctxp;
    for (int i = lane; i < t->n; i += lanes) {
        t->buf[i] = t->buf[i] * 1.000001f + 0.000001f;
    }
}

static void test_burn(void) {
    enum { N = 100000, RUNS = 50 };
    static float buf1[N], buf4[N];
    for (int i = 0; i < N; i++) buf1[i] = buf4[i] = (float)i / 1000.0f;

    si_tp *tp1 = si_tp_create(1);
    si_tp *tp4 = si_tp_create(3);
    burn_task t1 = { .buf = buf1, .n = N };
    burn_task t4 = { .buf = buf4, .n = N };
    for (int r = 0; r < RUNS; r++) {
        si_tp_run(tp1, burn_task_fn, &t1);
        si_tp_run(tp4, burn_task_fn, &t4);
    }
    for (int i = 0; i < N; i++) {
        CHECK(buf1[i] == buf4[i]);
    }
    si_tp_free(tp1);
    si_tp_free(tp4);
}

int main(void) {
    test_lane_invariance();
    test_rapid_dispatch();
    test_all_lanes_run();
    test_scratch_isolation();
    test_burn();
    TEST_REPORT("test_threadpool");
}
