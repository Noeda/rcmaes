#include <vector>
#include "libcmaes/cmaes.h"
#include "libcmaes/surrogates/rankingsvm.hpp"
#include "libcmaes/surrogates/rsvm_surr_strategy.hpp"
#include "surrcmaes_glue.h"
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>

using namespace libcmaes;

typedef struct scmaes_candidates_mvar {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    void* content;
    int nwaiters;
    int should_die;
} cmaes_candidates_mvar;

extern "C" {
    void cmaes_optimize(
            int noisy,
            int use_elitism,
            int use_surrogates,
            int algo,
            double* initial,
            double sigma,
            int lambda,
            uint64_t num_coords,
            double (*evaluate)(double*, int*, void*, int*),
            void (*iter)(void*),
            void (*tell_mvars)(void*, cmaes_candidates_mvar**, cmaes_candidates_mvar**, size_t),
            void (*wait_until_dead)(void*), // tells Rust to clean up, does not
                                            // return until clean up is done on
                                            // Rust side.
            void* userdata);
    cmaes_candidates_mvar* cmaes_make_candidates_mvar(void);
    void cmaes_mark_as_dead_mvar(cmaes_candidates_mvar* mvar);
    void cmaes_free_candidates_mvar(cmaes_candidates_mvar* mvar);
    void* cmaes_candidates_mvar_take(cmaes_candidates_mvar* mvar);
    void* cmaes_candidates_mvar_take_timeout(cmaes_candidates_mvar* mvar, int microseconds);
    int cmaes_candidates_mvar_give(cmaes_candidates_mvar* mvar, void* content);
    size_t cmaes_candidates_mvar_num_waiters(cmaes_candidates_mvar* mvar);
}

template<typename Tag, typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag) {
    return M;
  }
};

// Evil hack to access _run_status from CMASolutions
// Thanks to:
// https://bloglitb.blogspot.com/2011/12/access-to-private-members-safer.html
struct access_run_status {
    typedef int CMASolutions::*type;
    friend type get(access_run_status);
};

template struct Rob<access_run_status, &CMASolutions::_run_status>;

#define defconst(v) \
extern "C" { \
    int const_##v (void); \
} \
int const_##v (void) { return (v); };

defconst(CMAES_DEFAULT);
defconst(IPOP_CMAES);
defconst(BIPOP_CMAES);
defconst(aCMAES);
defconst(aIPOP_CMAES);
defconst(aBIPOP_CMAES);
defconst(sepCMAES);
defconst(sepIPOP_CMAES);
defconst(sepBIPOP_CMAES);
defconst(sepaCMAES);
defconst(sepaIPOP_CMAES);
defconst(sepaBIPOP_CMAES);
defconst(VD_CMAES);
defconst(VD_IPOP_CMAES);
defconst(VD_BIPOP_CMAES);

cmaes_candidates_mvar* cmaes_make_candidates_mvar(void) {
    cmaes_candidates_mvar* mvar = new cmaes_candidates_mvar();
    memset(mvar, 0, sizeof(cmaes_candidates_mvar));
    if (pthread_mutex_init(&mvar->lock, 0)) {
        fprintf(stderr, "pthread_mutex_init failed\n");
        exit(1);
    }
    if (pthread_cond_init(&mvar->cond, 0)) {
        pthread_mutex_destroy(&mvar->lock);
        fprintf(stderr, "pthread_cond_init failed\n");
        exit(1);
    }
    return mvar;
}

void cmaes_mark_as_dead_mvar(cmaes_candidates_mvar* mvar) {
    pthread_mutex_lock(&mvar->lock);
    mvar->should_die = 1;
    pthread_cond_broadcast(&mvar->cond);
    pthread_mutex_unlock(&mvar->lock);
}

void cmaes_free_candidates_mvar(cmaes_candidates_mvar* mvar) {
    pthread_mutex_lock(&mvar->lock);
    assert(mvar->nwaiters == 0);
    pthread_mutex_unlock(&mvar->lock);

    pthread_cond_destroy(&mvar->cond);
    pthread_mutex_destroy(&mvar->lock);
    delete mvar;
}

size_t cmaes_candidates_mvar_num_waiters(cmaes_candidates_mvar* mvar) {
    pthread_mutex_lock(&mvar->lock);
    size_t nwaiters = mvar->nwaiters;
    pthread_mutex_unlock(&mvar->lock);
    return nwaiters;
}

void* cmaes_candidates_mvar_take(cmaes_candidates_mvar* mvar) {
    pthread_mutex_lock(&mvar->lock);
    if (mvar->should_die) {
        pthread_mutex_unlock(&mvar->lock);
        return 0;
    }

    mvar->nwaiters++;
    while (mvar->content == 0 && !mvar->should_die) {
        pthread_cond_wait(&mvar->cond, &mvar->lock);
    }
    mvar->nwaiters--;
    if (mvar->should_die) {
        pthread_mutex_unlock(&mvar->lock);
        return 0;
    }

    void* content = mvar->content;
    mvar->content = 0;
    pthread_cond_signal(&mvar->cond);
    pthread_mutex_unlock(&mvar->lock);
    return content;
}

void* cmaes_candidates_mvar_take_timeout(cmaes_candidates_mvar* mvar, int microseconds) {
    pthread_mutex_lock(&mvar->lock);
    if (mvar->should_die) {
        pthread_mutex_unlock(&mvar->lock);
        return 0;
    }

    struct timespec ts;
    memset(&ts, 0, sizeof(ts));
    ts.tv_sec = (int64_t) microseconds / 1000000;
    ts.tv_nsec = ((int64_t) microseconds % 1000000) * 1000;

    mvar->nwaiters++;
    while (mvar->content == 0 && !mvar->should_die) {
        if (pthread_cond_timedwait(&mvar->cond, &mvar->lock, &ts) == ETIMEDOUT) {
            mvar->nwaiters--;
            pthread_mutex_unlock(&mvar->lock);
            return 0;
        }
    }
    mvar->nwaiters--;
    if (mvar->should_die) {
        pthread_mutex_unlock(&mvar->lock);
        return 0;
    }

    void* content = mvar->content;
    mvar->content = 0;
    pthread_cond_signal(&mvar->cond);
    pthread_mutex_unlock(&mvar->lock);
    return content;
}

int cmaes_candidates_mvar_give(cmaes_candidates_mvar* mvar, void* content) {
    assert(content);

    if (mvar->should_die) {
        return 0;
    }

    pthread_mutex_lock(&mvar->lock);
    if (mvar->should_die) {
        pthread_mutex_unlock(&mvar->lock);
        return 0;
    }

    mvar->nwaiters++;
    while (mvar->content != 0 && !mvar->should_die) {
        pthread_cond_wait(&mvar->cond, &mvar->lock);
    }
    mvar->nwaiters--;
    if (mvar->should_die) {
        pthread_mutex_unlock(&mvar->lock);
        return 0;
    }

    mvar->content = content;
    pthread_cond_signal(&mvar->cond);
    pthread_mutex_unlock(&mvar->lock);
    return 1;
}

void cmaes_optimize(
        int noisy,
        int use_elitism,
        int use_surrogates,
        int algo,
        double* initial,
        double sigma,
        int lambda,
        uint64_t num_coords,
        double (*evaluate)(double*, int*, void*, int*),
        void (*iter)(void*),
        void (*tell_mvars)(void*, cmaes_candidates_mvar**, cmaes_candidates_mvar**, size_t),
        void (*wait_until_dead)(void*), // tells Rust to clean up, does not
                                        // return until clean up is done on
                                        // Rust side.
        void* userdata)
{
    int uses_threaded_candidates = 0;
    if (tell_mvars) {
        uses_threaded_candidates = 1;
        assert(wait_until_dead);
    } else {
        assert(!wait_until_dead);
    }

    // lambda = number of offspring
    std::vector<cmaes_candidates_mvar*> candidates_mvars_outgoing;
    std::vector<cmaes_candidates_mvar*> candidates_mvars_incoming;
    for ( int i1 = 0; i1 < lambda; ++i1 ) {
        candidates_mvars_outgoing.push_back(cmaes_make_candidates_mvar());
        candidates_mvars_incoming.push_back(cmaes_make_candidates_mvar());
    }
    pthread_mutex_t candidates_lock;
    if (pthread_mutex_init(&candidates_lock, 0)) {
        fprintf(stderr, "pthread_mutex_init failed\n");
        exit(1);
    }

    if (tell_mvars) {
        tell_mvars(userdata, &candidates_mvars_outgoing[0], &candidates_mvars_incoming[0], candidates_mvars_outgoing.size());
    }

#define CLEANUP { pthread_mutex_destroy(&candidates_lock); \
    for (uint64_t i1 = 0; i1 < candidates_mvars_outgoing.size(); ++i1 ) { \
        cmaes_mark_as_dead_mvar(candidates_mvars_outgoing[i1]); \
        cmaes_mark_as_dead_mvar(candidates_mvars_incoming[i1]); \
    } \
    if (wait_until_dead) { wait_until_dead(userdata); } \
    for (uint64_t i1 = 0; i1 < candidates_mvars_outgoing.size(); ++i1 ) { \
        cmaes_free_candidates_mvar(candidates_mvars_outgoing[i1]); \
        cmaes_free_candidates_mvar(candidates_mvars_incoming[i1]); \
    } }

    std::vector<double> x0;
    for ( uint64_t i1 = 0; i1 < num_coords; ++i1 ) {
        x0.push_back(initial[i1]);
    }
    CMAParameters<> cmaparams(x0, sigma, lambda);

    volatile CMASolutions* sols = 0;
    volatile int should_stop = 0;

    size_t cand_idx = 0;

    FitFunc fit = [&uses_threaded_candidates, &cand_idx, &should_stop, &evaluate, &userdata, &sols, &candidates_mvars_outgoing, &candidates_mvars_incoming, &candidates_lock](const double* params, const int N) {
        ((void) N);
        int dumb = 0;
        int stop = 0;
        double result = 0.0;

        if (uses_threaded_candidates) {
            pthread_mutex_lock(&candidates_lock);
            cmaes_candidates_mvar* cvar_outgoing = candidates_mvars_outgoing[cand_idx];
            cmaes_candidates_mvar* cvar_incoming = candidates_mvars_incoming[cand_idx];
            cand_idx = (cand_idx + 1) % candidates_mvars_outgoing.size();
            pthread_mutex_unlock(&candidates_lock);
            if (!cmaes_candidates_mvar_give(cvar_outgoing, (void*) params)) {
                stop = 1;
            } else {
                void* content = NULL;
                if (!(content = cmaes_candidates_mvar_take(cvar_incoming))) {
                    stop = 1;
                } else {
                    memcpy(&result, &content, sizeof(double));
                }
            }
        } else {
            result = evaluate(const_cast<double*>(params), &dumb, userdata, &stop);
        }

        if (stop) {
            should_stop = 1;
            if (sols != 0) {
                (*sols).*get(access_run_status()) = -1;
            }
        }
        return result;
    };

    ProgressFunc<CMAParameters<>,CMASolutions> pfunc = [&should_stop, &iter, &userdata, &sols](const CMAParameters<> &cmaparams, const CMASolutions &cmasolutions)
    {
        ((void) cmaparams);
        if (sols == 0) {
            sols = (volatile CMASolutions*) &cmasolutions;
            if (should_stop) {
                (*sols).*get(access_run_status()) = -1;
            }
        }
        if (iter) {
            iter(userdata);
        }
        return 0;
    };

    cmaparams.set_algo(algo);
    cmaparams.set_mt_feval(true);
    if (noisy) {
        cmaparams.set_noisy();
    }
    if (use_elitism) {
        cmaparams.set_elitism(1);
    }

    std::vector<double> out;
    if ( use_surrogates ) {
        CMASolutions cmasols = surrcmaes<>(fit, cmaparams, pfunc);
        cmasols.sort_candidates();
        out = cmasols.best_candidate().get_x();
    } else {
        CMASolutions cmasols = cmaes<>(fit, cmaparams, pfunc);
        cmasols.sort_candidates();
        out = cmasols.best_candidate().get_x();
    }

    for ( uint64_t i1 = 0; i1 < num_coords; ++i1 ) {
        initial[i1] = out[i1];
    }

    CLEANUP;
}

