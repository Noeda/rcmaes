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
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList};

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
}

impl<T: Vectorizable> PyCMAESItem<T> {
    pub fn item(&self) -> &T {
        &self.item
    }

    // TODO: according to cma documentation, passing candidates manually to CMA can have negative
    // consequences when active CMA is used. We could take that into account and properly inject
    // solutions the sanctioned way.
    // (https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html)
    pub fn set_item(&mut self, item: T) {
        (self.vec, _) = item.to_vec();
        self.item = item;
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
    optimizer: PyCMAESOptimizer,
    population_size: Option<usize>,
}

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub enum PyCMAESOptimizer {
    CMA,    // Regular CMA-ES that you get from import CMA
    VkDCMA, // VkD CMA-ES
}

impl PyCMAESOptimizer {
    fn to_string(&self) -> &'static str {
        match self {
            PyCMAESOptimizer::CMA => "CMA",
            PyCMAESOptimizer::VkDCMA => "VkDCMA",
        }
    }
}

impl PyCMAESSettings {
    pub fn new() -> Self {
        PyCMAESSettings {
            sigma: 1.0,
            optimizer: PyCMAESOptimizer::CMA,
            active: true,
            population_size: None,
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
}

impl<T: Vectorizable + Clone> PyCMAES<T> {
    pub fn new(initial: &T, pycmaes_settings: PyCMAESSettings) -> Self {
        let (vec, ctx) = initial.to_vec();
        let dimension = vec.len();

        Python::with_gil(|py| {
            // Call make_cmaes in PYCMAES_MODULE
            let cmaes_module =
                PyModule::from_code(py, PYCMAES_CODE, "pycmaes.py", "pycmaes").unwrap();
            let make_cmaes = cmaes_module.getattr("make_cmaes").unwrap();

            let optimizer_name = format!("{}", pycmaes_settings.optimizer.to_string());

            let pyvec = PyList::new(py, &vec);

            let optimizer = make_cmaes
                .call1((
                    pyvec,
                    pycmaes_settings.sigma,
                    optimizer_name,
                    pycmaes_settings.population_size,
                    pycmaes_settings.active,
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
                });
            }
            items
        })
    }

    pub fn tell(&mut self, candidate: Vec<PyCMAESItem<T>>) {
        Python::with_gil(|py| {
            let tell = self.cmaes_module.getattr(py, "tell").unwrap();
            let pyvec = PyList::empty(py);
            for item in candidate {
                let pyitem = PyList::new(py, &item.vec);
                let pyscore = PyFloat::new(py, item.score);
                let tup = PyList::new(py, &[PyObject::from(pyitem), PyObject::from(pyscore)]);
                pyvec.append(tup).unwrap();
            }
            tell.call1(py, (self.optimizer.clone_ref(py), pyvec))
                .unwrap();
        })
    }
}

// Python code for all this glue
const PYCMAES_CODE: &str = r#"
import cma
from cma import restricted_gaussian_sampler as rgs
import numpy as np

def make_cmaes(initial, sigma, optimizer_name, population_size=None, active=None):
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

    result = cma.CMAEvolutionStrategy(initial, sigma, opts)
    return result

def ask(optimizer):
    if optimizer.stop():
        return []
    result = optimizer.ask()
    result = list(map(list, result))
    return result

def tell(optimizer, solutions):
    firsts = [x[0] for x in solutions]
    seconds = [x[1] for x in solutions]
    optimizer.tell(firsts, seconds)

def population_size(optimizer):
    return optimizer.popsize
"#;

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

    fn test_run(optim: PyCMAESOptimizer) {
        let mut cmaes = PyCMAES::new(
            &TestVector { x: 1.9, y: -3.0 },
            PyCMAESSettings::new().optimizer(optim),
        );

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
            for mut item in items.into_iter() {
                // Rosenbrock function
                item.set_score(
                    (A - item.item().x).powi(2)
                        + B * (item.item().y - item.item().x.powi(2)).powi(2),
                );
                solutions.push(item);
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
        test_run(PyCMAESOptimizer::CMA);
        test_run(PyCMAESOptimizer::VkDCMA);
    }
}
