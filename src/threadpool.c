/* siphon: persistent worker pool (see include/siphon/threadpool.h) */
#include "siphon/threadpool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

typedef struct {
    _Atomic uint64_t gen;        /* bumped by each dispatch            */
    _Atomic int64_t  done;       /* lanes finished in current gen      */
    si_task_fn       fn;         /* current task (constant during run) */
    void            *ctx;
    int              n_lanes;
    _Atomic bool     shutdown;
    /* parking */
    pthread_mutex_t  mu;
    pthread_cond_t   cv;
    pthread_t       *tid;
    int              n_workers; /* n_lanes - 1                        */
} si_tp_impl;

enum { SI_SPIN_ROUNDS = 60000 };

static void worker_loop(si_tp_impl *p, int lane) {
    uint64_t seen = 0;
    for (;;) {
        /* spin phase: catch the next dispatch without a syscall */
        uint64_t spins = 0;
        uint64_t gen;
        for (;;) {
            gen = atomic_load_explicit(&p->gen, memory_order_acquire);
            if (gen != seen) break;
            if (atomic_load_explicit(&p->shutdown, memory_order_relaxed)) return;
            if (++spins > (uint64_t)SI_SPIN_ROUNDS) {
                /* park; the gen re-check under the lock closes the race
                 * with a dispatch that fired just before we locked */
                pthread_mutex_lock(&p->mu);
                gen = atomic_load_explicit(&p->gen, memory_order_acquire);
                if (gen == seen && !atomic_load_explicit(&p->shutdown, memory_order_relaxed)) {
                    pthread_cond_wait(&p->cv, &p->mu);
                }
                pthread_mutex_unlock(&p->mu);
                gen = atomic_load_explicit(&p->gen, memory_order_acquire);
                break;
            }
        }
        if (gen == seen) continue; /* spurious wake or shutdown */
        seen = gen;
        p->fn(p->ctx, lane, p->n_lanes);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
}

struct worker_arg {
    si_tp_impl *p;
    int lane;
};

static void *worker_entry(void *argp) {
    struct worker_arg *w = argp;
    worker_loop(w->p, w->lane);
    free(w);
    return NULL;
}

si_tp *si_tp_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    si_tp_impl *p = si_xcalloc(1, sizeof *p);
    atomic_init(&p->gen, 0);
    atomic_init(&p->done, 0);
    atomic_init(&p->shutdown, false);
    p->n_lanes = n_threads;
    p->n_workers = n_threads - 1;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    if (p->n_workers > 0) {
        p->tid = si_xmalloc(sizeof(pthread_t) * (size_t)p->n_workers);
        for (int i = 0; i < p->n_workers; i++) {
            struct worker_arg *w = si_xmalloc(sizeof *w);
            w->p = p;
            w->lane = i + 1;
            if (pthread_create(&p->tid[i], NULL, worker_entry, w) != 0) {
                /* degrade gracefully: run the lanes we managed to start */
                free(w);
                p->tid[i] = pthread_self(); /* never joined; marker */
                p->n_workers = i;
                p->n_lanes = i + 1;
                break;
            }
        }
    }
    return (si_tp *)p;
}

void si_tp_free(si_tp *tp) {
    if (!tp) return;
    si_tp_impl *p = (si_tp_impl *)tp;
    if (p->n_workers > 0 && p->tid) {
        atomic_store_explicit(&p->shutdown, true, memory_order_relaxed);
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
        for (int i = 0; i < p->n_workers; i++) {
            pthread_join(p->tid[i], NULL);
        }
        free(p->tid);
    }
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    free(p);
}

int si_tp_nlanes(const si_tp *tp) {
    return ((si_tp_impl *)tp)->n_lanes;
}

void si_tp_run(si_tp *tp, si_task_fn fn, void *ctx) {
    si_tp_impl *p = (si_tp_impl *)tp;
    if (p->n_lanes == 1) {
        fn(ctx, 0, 1);
        return;
    }
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    p->fn = fn;
    p->ctx = ctx;
    /* release the workers; the acquire on gen in worker_loop pairs with
     * this release so fn/ctx are visible before any lane starts */
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
    pthread_mutex_lock(&p->mu);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);

    /* lane 0 runs on the calling thread */
    fn(ctx, 0, p->n_lanes);
    atomic_fetch_add_explicit(&p->done, 1, memory_order_release);

    /* wait for the other lanes: spin, then poll with short sleeps */
    const int64_t want = p->n_lanes;
    uint64_t spins = 0;
    while (atomic_load_explicit(&p->done, memory_order_acquire) < want) {
        if (++spins > (uint64_t)SI_SPIN_ROUNDS) {
            struct timespec ts = { 0, 100000 }; /* 100 us */
            nanosleep(&ts, NULL);
        }
    }
}
