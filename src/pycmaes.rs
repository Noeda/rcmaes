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

pub struct PyCMAESItem<T> {
    item: T,
    vec: Vec<f64>,
    score: f64,
}

impl<T> PyCMAESItem<T> {
    pub fn item(&self) -> &T {
        &self.item
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
    optimizer: PyCMAESOptimizer,
}

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub enum PyCMAESOptimizer {
    CMA,    // Regular CMA-ES that you get from import CMA
    SepCMA, // Separable CMA-ES
}

impl PyCMAESOptimizer {
    fn to_string(&self) -> &'static str {
        match self {
            PyCMAESOptimizer::CMA => "CMA",
            PyCMAESOptimizer::SepCMA => "SepCMA",
        }
    }
}

impl PyCMAESSettings {
    pub fn new() -> Self {
        PyCMAESSettings {
            sigma: 1.0,
            optimizer: PyCMAESOptimizer::CMA,
        }
    }

    pub fn optimizer(mut self, optimizer: PyCMAESOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
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
                .call1((pyvec, pycmaes_settings.sigma, optimizer_name))
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

    pub fn ask(&self) -> PyCMAESItem<T> {
        Python::with_gil(|py| {
            let ask = self.cmaes_module.getattr(py, "ask").unwrap();
            let pyvec = ask.call1(py, (self.optimizer.clone_ref(py),)).unwrap();
            let vec: Vec<f64> = pyvec.extract(py).unwrap();
            PyCMAESItem {
                item: T::from_vec(&vec, &self.ctx),
                vec,
                score: 0.0,
            }
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
import cmaes
import numpy as np

def make_cmaes(initial, sigma, optimizer_name, bounds=None):
    if bounds is not None:
        np_bounds = np.concatenate(
            [
                np.tile([-np.inf, np.inf], (len(initial), 1)),
            ]
        )
        for idx in range(len(initial)):
            if len(bounds) == 1:
                np_bounds[idx, 0] = bounds[0][0]
                np_bounds[idx, 1] = bounds[0][1]
            else:
                np_bounds[idx, 0] = bounds[idx][0]
                np_bounds[idx, 1] = bounds[idx][1]

        return getattr(cmaes, optimizer_name)(np.array(initial), sigma, bounds=np_bounds, steps=np.zeros(len(initial)))
    else:
        return getattr(cmaes, optimizer_name)(np.array(initial), sigma)

def ask(optimizer):
    return list(optimizer.ask())

def tell(optimizer, solutions):
    for sol in solutions:
        sol[0] = np.array(sol[0])
    optimizer.tell(solutions)

def population_size(optimizer):
    return optimizer.population_size
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

        for idx in 0..10000 {
            let mut solutions = vec![];
            for _ in 0..cmaes.population_size() {
                let mut item = cmaes.ask();
                // Rosenbrock function
                item.set_score(
                    (A - item.item().x).powi(2)
                        + B * (item.item().y - item.item().x.powi(2)).powi(2),
                );
                solutions.push(item);
            }
            if idx == 999 {
                let item = solutions[0].item();
                // Make sure the item is close enough
                assert!((item.x - expected_solution.0).abs() < 0.1);
                assert!((item.y - expected_solution.1).abs() < 0.1);
            }
            cmaes.tell(solutions);
        }
    }

    #[test]
    fn test_pycmaes() {
        pyo3::prepare_freethreaded_python();
        test_run(PyCMAESOptimizer::CMA);
    }
}
