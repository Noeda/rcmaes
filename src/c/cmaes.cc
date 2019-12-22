#include <vector>
#include "libcmaes/cmaes.h"
#include "libcmaes/surrcmaes.h"
#include "libcmaes/surrogates/rankingsvm.hpp"
#include "libcmaes/surrogates/rsvm_surr_strategy.hpp"
#include <stdint.h>

using namespace libcmaes;

extern "C" {
    void cmaes_optimize(int use_elitism, int use_surrogates, int algo, double* initial, double sigma, int lambda, uint64_t num_coords, double (*evaluate)(double*, int*, void*), void (*iterator)(void*), void* userdata);
}

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

void cmaes_optimize(int use_elitism, int use_surrogates, int algo, double* initial, double sigma, int lambda, uint64_t num_coords, double (*evaluate)(double*, int*, void*), void (*iter)(void*), void* userdata)
{
    FitFunc fit = [&evaluate, &userdata](const double* params, const int N) {
        ((void) N);
        int dumb = 0;
        return evaluate(const_cast<double*>(params), &dumb, userdata);
    };

    ProgressFunc<CMAParameters<>,CMASolutions> pfunc = [&iter, &userdata](const CMAParameters<> &cmaparams, const CMASolutions &cmasolutions)
    {
        ((void) cmaparams);
        ((void) cmasolutions);
        iter(userdata);
        return 0;
    };

    std::vector<double> x0;
    for ( uint64_t i1 = 0; i1 < num_coords; ++i1 ) {
        x0.push_back(initial[i1]);
    }

    CMAParameters<> cmaparams(x0, sigma, lambda);
    cmaparams.set_algo(algo);
    cmaparams.set_mt_feval(true);
    if (use_elitism) {
        cmaparams.set_elitism(1);
    }

    std::vector<double> out;
    if ( use_surrogates ) {
        CMASolutions cmasols = surrcmaes<>(fit, cmaparams);
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

