#include <vector>
#include "libcmaes/cmaes.h"
#include "libcmaes/surrogates/rankingsvm.hpp"
#include "libcmaes/surrogates/rsvm_surr_strategy.hpp"
#include "surrcmaes_glue.h"
#include <stdint.h>

using namespace libcmaes;

extern "C" {
    void cmaes_optimize(int noisy, int elitism_reevaluate, int use_elitism, int use_surrogates, int algo, double* initial, double sigma, int lambda, uint64_t num_coords, double (*evaluate)(double*, int*, void*, int*), void (*iterator)(void*), void* userdata);
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

void cmaes_optimize(int noisy, int elitism_reevaluate, int use_elitism, int use_surrogates, int algo, double* initial, double sigma, int lambda, uint64_t num_coords, double (*evaluate)(double*, int*, void*, int*), void (*iter)(void*), void* userdata)
{
    std::vector<double> x0;
    for ( uint64_t i1 = 0; i1 < num_coords; ++i1 ) {
        x0.push_back(initial[i1]);
    }
    CMAParameters<> cmaparams(x0, sigma, lambda);

    volatile CMASolutions* sols = 0;
    volatile int should_stop = 0;

    FitFunc fit = [&should_stop, &evaluate, &userdata, &sols](const double* params, const int N) {
        ((void) N);
        int dumb = 0;
        int stop = 0;
        double result = evaluate(const_cast<double*>(params), &dumb, userdata, &stop);
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
        iter(userdata);
        return 0;
    };

    cmaparams.set_algo(algo);
    cmaparams.set_mt_feval(true);
    if (noisy) {
        cmaparams.set_noisy();
    }
    if (use_elitism) {
        cmaparams.set_elitism(1);
        if (elitism_reevaluate) {
            cmaparams.set_revaluate_elite(1);
        }
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
}

