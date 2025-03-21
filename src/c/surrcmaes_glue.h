#pragma once

namespace libcmaes
{
  template <class TGenoPheno=GenoPheno<NoBoundStrategy> >
  CMASolutions CMAES_EXPORT surrcmaes_rsvm(FitFunc &func,
			 CMAParameters<TGenoPheno> &parameters,
                         ProgressFunc<CMAParameters<TGenoPheno>,CMASolutions> &pfunc=CMAStrategy<CovarianceUpdate,TGenoPheno>::_defaultPFunc, int niters=0)
    {
      switch(parameters.get_algo())
	{
	case CMAES_DEFAULT:
	{
	  ESOptimizer<RSVMSurrogateStrategy<CMAStrategy,CovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case IPOP_CMAES:
	{
	  ESOptimizer<RSVMSurrogateStrategy<IPOPCMAStrategy,CovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case BIPOP_CMAES:
	{
	  ESOptimizer<RSVMSurrogateStrategy<BIPOPCMAStrategy,CovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case aCMAES:
	{
	  ESOptimizer<RSVMSurrogateStrategy<CMAStrategy,ACovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case aIPOP_CMAES:
	{
	  ESOptimizer<RSVMSurrogateStrategy<IPOPCMAStrategy,ACovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case aBIPOP_CMAES:
	{
	  ESOptimizer<RSVMSurrogateStrategy<BIPOPCMAStrategy,ACovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case sepCMAES:
	{
	  parameters.set_sep();
	  ESOptimizer<RSVMSurrogateStrategy<CMAStrategy,CovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case sepIPOP_CMAES:
	{
	  parameters.set_sep();
	  ESOptimizer<RSVMSurrogateStrategy<IPOPCMAStrategy,CovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case sepBIPOP_CMAES:
	{
	  parameters.set_sep();
	  ESOptimizer<RSVMSurrogateStrategy<BIPOPCMAStrategy,CovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case sepaCMAES:
	{
	  parameters.set_sep();
	  ESOptimizer<RSVMSurrogateStrategy<CMAStrategy,ACovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case sepaIPOP_CMAES:
	{
	  parameters.set_sep();
	  ESOptimizer<RSVMSurrogateStrategy<IPOPCMAStrategy,ACovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case sepaBIPOP_CMAES:
	{
	  parameters.set_sep();
	  ESOptimizer<RSVMSurrogateStrategy<BIPOPCMAStrategy,ACovarianceUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case VD_CMAES:
	{
	  parameters.set_vd();
	  ESOptimizer<RSVMSurrogateStrategy<CMAStrategy,VDCMAUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case VD_IPOP_CMAES:
	{
	  parameters.set_vd();
	  ESOptimizer<RSVMSurrogateStrategy<IPOPCMAStrategy,VDCMAUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	case VD_BIPOP_CMAES:
	{
	  parameters.set_vd();
	  ESOptimizer<RSVMSurrogateStrategy<BIPOPCMAStrategy,VDCMAUpdate>,CMAParameters<>> optim(func,parameters);
          if (niters > 0) {
            optim._rsvm_iter = niters;
          }
          optim.set_progress_func(pfunc);
	  optim.optimize();
	  return optim.get_solutions();
	}
	default:
	return CMASolutions();
	}
    }
}
