/*
 * This part integrates with the Python cmaes library, that has more recent features
 * than the C++ libcmaes.
 *
 * Maybe eventually this becomes the main way to use CMA-ES in this library, unless libcmaes starts
 * getting new features.
 */

/*
 * The interface mimics the Python cmaes interface. which is described as an "ask-and-tell" method.
 *
 * You create a CMA-ES object, and then ask it to give you a new candidate to evaluate and score.
 * You can request many of them at once, and then evaluate them in parallel, and pass them back to
 * the CMA-ES object.
 */

use crate::vectorizable::Vectorizable;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList};
use std::ffi::CStr;

#[derive(Debug)]
pub struct PyCMAES<T: Vectorizable> {
    optimizer: PyObject,
    cmaes_module: Py<PyModule>,
    _dimension: usize,
    ctx: T::Context,
    // Phantom
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Clone, Debug)]
pub struct PyCMAESItem<T> {
    item: T,
    vec: Vec<f64>,
    score: f64,
    injected: Option<Vec<f64>>,
}

impl<T: Vectorizable> PyCMAESItem<T> {
    pub fn item(&self) -> &T {
        &self.item
    }

    pub fn set_item(&mut self, item: T) {
        let (vec, _) = item.to_vec();
        // According to cma documentation, passing candidates manually to CMA can have negative
        // consequences when active CMA is used. We want to use a inject() call to do it properly.
        // PyCMA still wants the original solution, so we keep it around.
        // (https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html)
        self.injected = Some(vec);
    }

    pub fn set_score(&mut self, score: f64) {
        self.score = score;
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct PyCMAESSettings {
    sigma: f64,
    active: bool,
    use_adapt_sigma_tpa: bool,
    optimizer: PyCMAESOptimizer,
    population_size: Option<usize>,
    never_stop: bool,
}

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub enum PyCMAESOptimizer {
    CMA,         // Regular CMA-ES that you get from import CMA
    VkDCMA,      // VkD CMA-ES
    CMAOther,    // CMA-ES from CyberAgentAILab/cmaes instead of 'cma'
    LRACMAOther, // LRACMA-ES from CyberAgentAILab/cmaes instead of 'cma'
    XNESOther,
    DXNESICOther,
}

impl PyCMAESOptimizer {
    fn to_string(&self) -> &'static str {
        match self {
            PyCMAESOptimizer::CMA => "CMA",
            PyCMAESOptimizer::VkDCMA => "VkDCMA",
            PyCMAESOptimizer::CMAOther => "CMAOther",
            PyCMAESOptimizer::LRACMAOther => "LRACMAOther",
            PyCMAESOptimizer::XNESOther => "XNESOther",
            PyCMAESOptimizer::DXNESICOther => "DXNESICOther",
        }
    }
}

impl PyCMAESSettings {
    pub fn new() -> Self {
        PyCMAESSettings {
            sigma: 1.0,
            optimizer: PyCMAESOptimizer::CMA,
            active: true,
            use_adapt_sigma_tpa: false,
            population_size: None,
            never_stop: false,
        }
    }

    pub fn population_size(mut self, pop_size: usize) -> Self {
        self.population_size = Some(pop_size);
        self
    }

    pub fn optimizer(mut self, optimizer: PyCMAESOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    pub fn use_adapt_sigma_tpa(mut self, use_adapt_sigma_tpa: bool) -> Self {
        self.use_adapt_sigma_tpa = use_adapt_sigma_tpa;
        self
    }

    pub fn never_stop(mut self, never_stop: bool) -> Self {
        self.never_stop = never_stop;
        self
    }
}

impl<T: Vectorizable + Clone> PyCMAES<T> {
    pub fn new(initial: &T, pycmaes_settings: PyCMAESSettings) -> Self {
        let (vec, ctx) = initial.to_vec();
        let dimension = vec.len();

        Python::with_gil(|py| {
            // Call make_cmaes in PYCMAES_MODULE
            let cmaes_module =
                PyModule::from_code(py, PYCMAES_CODE, c_str!("pycmaes.py"), c_str!("pycmaes"))
                    .unwrap();
            let make_cmaes = cmaes_module.getattr("make_cmaes").unwrap();

            let optimizer_name = format!("{}", pycmaes_settings.optimizer.to_string());

            let pyvec = PyList::new(py, &vec).unwrap();

            let optimizer = make_cmaes
                .call1((
                    pyvec,
                    pycmaes_settings.sigma,
                    optimizer_name,
                    pycmaes_settings.population_size,
                    pycmaes_settings.active,
                    pycmaes_settings.use_adapt_sigma_tpa,
                    pycmaes_settings.never_stop,
                ))
                .unwrap();

            PyCMAES {
                cmaes_module: cmaes_module.into(),
                optimizer: optimizer.into(),
                ctx,
                _dimension: dimension,
                _phantom: std::marker::PhantomData,
            }
        })
    }

    pub fn population_size(&self) -> usize {
        Python::with_gil(|py| {
            let population_size = self.cmaes_module.getattr(py, "population_size").unwrap();
            population_size
                .call1(py, (self.optimizer.clone_ref(py),))
                .unwrap()
                .extract(py)
                .unwrap()
        })
    }

    pub fn ask(&self) -> Vec<PyCMAESItem<T>> {
        Python::with_gil(|py| {
            let ask = self.cmaes_module.getattr(py, "ask").unwrap();
            let pyvec = ask
                .call1(py, (self.optimizer.clone_ref(py),))
                .unwrap()
                .extract::<Vec<Vec<f64>>>(py)
                .unwrap();

            let mut items = Vec::new();
            for vec in pyvec {
                let item = T::from_vec(&vec.clone(), &self.ctx);
                items.push(PyCMAESItem {
                    item,
                    vec,
                    score: 0.0,
                    injected: None,
                });
            }
            items
        })
    }

    pub fn tell(&mut self, candidate: Vec<PyCMAESItem<T>>) {
        Python::with_gil(|py| {
            let tell = self.cmaes_module.getattr(py, "tell").unwrap();
            let pyvec = PyList::empty(py);
            for item in candidate.iter() {
                let pyitem = PyList::new(py, &item.vec).unwrap();
                let pyscore = PyFloat::new(py, item.score);
                let tup =
                    PyList::new(py, &[PyObject::from(pyitem), PyObject::from(pyscore)]).unwrap();
                pyvec.append(tup).unwrap();
            }
            tell.call1(py, (self.optimizer.clone_ref(py), pyvec))
                .unwrap();
            let pyvec = PyList::empty(py);
            let inject = self.cmaes_module.getattr(py, "inject").unwrap();
            let mut has_injections = false;
            for candidate in candidate.iter() {
                if let Some(ref injected_vec) = candidate.injected {
                    has_injections = true;
                    pyvec
                        .append(PyList::new(py, injected_vec).unwrap())
                        .unwrap();
                }
            }
            if has_injections {
                inject
                    .call1(py, (self.optimizer.clone_ref(py), pyvec))
                    .unwrap();
            }
        })
    }

    pub fn sigma(&self) -> f64 {
        Python::with_gil(|py| {
            let get_sigma = self.cmaes_module.getattr(py, "get_sigma").unwrap();
            let sigma = get_sigma
                .call1(py, (self.optimizer.clone_ref(py),))
                .unwrap()
                .extract::<f64>(py)
                .unwrap();
            sigma
        })
    }
}

// Python code for all this glue
const PYCMAES_CODE: &CStr = c_str!(
    r#"
import sys
import os

# Respect virtualenv, if set. The interpreter we are running as might be totally different though I suppose.
if os.environ.get('VIRTUAL_ENV', None):
    # add search paths
    SITE_PACKAGES_ADDITIONAL = os.path.join(os.environ['VIRTUAL_ENV'], 'lib', 'python' + f'{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
    sys.path.insert(0, SITE_PACKAGES_ADDITIONAL)

import cma
import cma.sigma_adaptation
from cma import restricted_gaussian_sampler as rgs
import numpy as np
import numpy.linalg as la
import math

# This is an entirely different CMA-ES implementation, also in Python.
try:
    import cmaes as cma_that_other_library
except ImportError:
    cma_that_other_library = None

class Opt:
    def __init__(self, cmaes):
        self.cmaes = cmaes
        self.never_stop = False
        self.is_other_cma = False
        self.dead = False

    def should_stop(self):
        if self.dead:
            return True
        if not self.never_stop:
            return False

        if getattr(self.cmaes, 'should_stop', None) is None:
            return self.cmaes.should_stop()
        elif getattr(self.cmaes, 'stop', None) is None:
            return self.cmaes.stop()
        else:
            assert False, "Optimizer has no stop() or should_stop()"

    def ask(self):
        if not self.is_other_cma:
            cands = self.cmaes.ask()
            return list(map(list, cands))
        else:
            # The other library doesn't give all candidates right away
            result = []
            for _ in range(self.cmaes.population_size):
                item = list(self.cmaes.ask())
                for x in item:
                    if math.isnan(x):
                        self.dead = True
                        return []
                result.append(list(self.cmaes.ask()))
            return result

def make_cmaes(initial, sigma, optimizer_name, population_size=None, active=None, use_adapt_sigma_tpa=False, never_stop=False):
    if optimizer_name == 'CMAOther' or optimizer_name == 'LRACMAOther' or optimizer_name == 'XNESOther' or optimizer_name == 'DXNESICOther':
        if use_adapt_sigma_tpa:
            raise ValueError("Cannot use adapt sigma TPA with CMAOther")

        use_lra = optimizer_name == 'LRACMAOther'
        return make_cmaes_other(optimizer_name, initial, sigma, population_size, never_stop, use_lra)

    opts = cma.CMAOptions()
    if active is not None:
        opts.set('CMA_active', active)
    if optimizer_name == 'CMA':
        pass
    elif optimizer_name == 'VkDCMA':
        opts = rgs.GaussVDSampler.extend_cma_options(opts)
    else:
        raise ValueError("Unknown optimizer: " + optimizer_name)
    if population_size is not None:
        opts.set('popsize', population_size)

    if use_adapt_sigma_tpa:
        opts.set('AdaptSigma', cma.sigma_adaptation.CMAAdaptSigmaTPA)

    result = Opt(cmaes=cma.CMAEvolutionStrategy(initial, sigma, opts))
    result.never_stop = never_stop
    return result

def make_cmaes_other(optimizer_name, initial, sigma, population_size, never_stop, use_lra):
    if optimizer_name == 'XNESOther':
        result = Opt(cmaes=cma_that_other_library.XNES(mean=initial, sigma=sigma, population_size=population_size))
    elif optimizer_name == 'DXNESICOther':
        result = Opt(cmaes=cma_that_other_library.DXNESIC(mean=initial, sigma=sigma, population_size=population_size))
    else:
        result = Opt(cmaes=cma_that_other_library.CMA(mean=initial, sigma=sigma, population_size=population_size, lr_adapt=use_lra))
    result.never_stop = never_stop
    result.is_other_cma = True
    return result

def ask(optimizer):
    if optimizer.should_stop():
        return []
    return optimizer.ask()

def tell(optimizer, solutions):
    if optimizer.is_other_cma:
        pr = []
        for x, y in solutions:
            pr.append((np.array(x), y))
        try:
            optimizer.cmaes.tell(pr)
        except OverflowError:
            optimizer.dead = True
        except la.LinAlgError:
            optimizer.dead = True
    else:
        firsts = [x[0] for x in solutions]
        seconds = [x[1] for x in solutions]
        optimizer.cmaes.tell(firsts, seconds)

def inject(optimizer, solutions):
    optimizer.cmaes.inject(solutions)

def population_size(optimizer):
    return optimizer.cmaes.popsize

def get_sigma(optimizer):
    if optimizer.is_other_cma:
        return 1
    else:
        return optimizer.cmaes.sigma
"#
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorizable::Vectorizable;

    #[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
    struct TestVector {
        x: f64,
        y: f64,
    }

    impl Vectorizable for TestVector {
        type Context = ();

        fn to_vec(&self) -> (Vec<f64>, ()) {
            (vec![self.x, self.y], ())
        }

        fn from_vec(vec: &[f64], _ctx: &Self::Context) -> Self {
            TestVector {
                x: vec[0],
                y: vec[1],
            }
        }
    }

    fn test_run(optim: PyCMAESOptimizer, inject: bool, use_adapt_sigma_tpa: bool) {
        let mut settings = PyCMAESSettings::new().optimizer(optim).population_size(30);
        if use_adapt_sigma_tpa {
            settings = settings.use_adapt_sigma_tpa(true);
        }

        let mut cmaes = PyCMAES::new(&TestVector { x: 1.9, y: -3.0 }, settings);

        const A: f64 = 2.5;
        const B: f64 = 100.0;

        let expected_solution = (A, A.powi(2));

        let mut solutions = vec![];
        for _idx in 0..10000 {
            let items = cmaes.ask();
            if items.len() == 0 {
                break;
            }

            solutions.clear();
            for (idx, mut item) in items.into_iter().enumerate() {
                let score = (A - item.item().x).powi(2)
                    + B * (item.item().y - item.item().x.powi(2)).powi(2);
                // Rosenbrock function
                item.set_score(score);
                if inject && (idx == 12 || idx == 13) && _idx > 50 {
                    item.set_item(TestVector { x: 1.3, y: -1.2 });
                }
                solutions.push(item);
                let _ = cmaes.sigma();
            }
            cmaes.tell(solutions.clone());
        }
        let item = solutions[0].item();
        // Make sure the item is close enough
        assert!((item.x - expected_solution.0).abs() < 0.1);
        assert!((item.y - expected_solution.1).abs() < 0.1);
    }

    #[test]
    fn test_pycmaes() {
        pyo3::prepare_freethreaded_python();
        test_run(PyCMAESOptimizer::CMA, false, false);
        test_run(PyCMAESOptimizer::VkDCMA, false, false);
        test_run(PyCMAESOptimizer::CMA, false, true);
        test_run(PyCMAESOptimizer::VkDCMA, false, true);
    }

    #[test]
    fn test_cmaes_other() {
        pyo3::prepare_freethreaded_python();
        test_run(PyCMAESOptimizer::CMAOther, false, false);
    }

    #[test]
    fn test_cmaes_lracma_other() {
        pyo3::prepare_freethreaded_python();
        test_run(PyCMAESOptimizer::LRACMAOther, false, false);
    }

    #[test]
    fn test_cmaes_xnes_other() {
        pyo3::prepare_freethreaded_python();
        test_run(PyCMAESOptimizer::XNESOther, false, false);
    }

    #[test]
    fn test_cmaes_dxnesic_other() {
        pyo3::prepare_freethreaded_python();
        test_run(PyCMAESOptimizer::DXNESICOther, false, false);
    }

    #[test]
    fn test_inject() {
        pyo3::prepare_freethreaded_python();
        test_run(PyCMAESOptimizer::VkDCMA, true, false);
        test_run(PyCMAESOptimizer::CMA, true, false);
        test_run(PyCMAESOptimizer::VkDCMA, true, true);
        test_run(PyCMAESOptimizer::CMA, true, true);
    }
}
