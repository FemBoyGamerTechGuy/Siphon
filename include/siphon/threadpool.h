/* siphon: persistent worker pool
 *
 * One pool per engine; workers are created once and reused for every
 * parallel section. A run() dispatch wakes them, they spin briefly for
 * latency-critical reuse, then park on a condition variable until the
 * next dispatch (spin-then-park).
 *
 * Lane invariance: the task receives (lane, n_lanes) and must partition
 * its work deterministically by those two numbers only — contiguous
 * row chunks or block-aligned strides, never work stealing. Combined
 * with complete per-row computations this makes results independent of
 * the thread count, which the tests prove by construction.
 */
#ifndef SIPHON_THREADPOOL_H
#define SIPHON_THREADPOOL_H

#include "siphon/common.h"

typedef struct si_tp si_tp;

/* Task: execute the slice of work owned by `lane` out of `n_lanes`. */
typedef void (*si_task_fn)(void *ctx, int lane, int n_lanes);

/* n_threads >= 1. 1 creates no worker threads: run() executes inline. */
si_tp *si_tp_create(int n_threads);
void si_tp_free(si_tp *tp);

/* Number of lanes the pool will dispatch (== requested threads). */
int si_tp_nlanes(const si_tp *tp);

/* Dispatch fn to all lanes and block until every lane finished.
 * Re-entrant per pool is NOT supported (one dispatch at a time). */
void si_tp_run(si_tp *tp, si_task_fn fn, void *ctx);

#endif /* SIPHON_THREADPOOL_H */
