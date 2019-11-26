extern crate libc;

mod raw;

use libc::{c_double, c_int, c_void};
use std::sync::{Arc, Mutex, RwLock};

/// Trait for things you can turn into vector. Things you want to optimize with CMA-ES need to
/// implement this.
///
/// This trait includes a Context type that can be used to pass auxiliary information to rebuild
/// the structure.
///
/// Note that this library assumes that a context obtained from one instance can be applied safely
/// to some copy of the original instance safely.
pub trait Vectorizable {
    type Context;

    fn to_vec(&self) -> (Vec<f64>, Self::Context);
    fn from_vec(vec: &[f64], ctx: &Self::Context) -> Self;
}

impl Vectorizable for Vec<f64> {
    type Context = ();

    fn to_vec(&self) -> (Vec<f64>, Self::Context) {
        (self.clone(), ())
    }
    fn from_vec(vec: &[f64], _: &Self::Context) -> Self {
        vec.to_owned()
    }
}

/// This enumerates each variant of CMA-ES algorithm supported by libcmaes.
#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub enum CMAESAlgo {
    Default,
    IPOP,
    BIPOP,
    ACMAES,
    AIPOP,
    ABIPOP,
    SEP,
    SEPIPOP,
    SEPBIPOP,
    SEPA,
    SEPAIPOP,
    SEPABIPOP,
    VD,
    VDIPOP,
    VDBIPOP,
}

impl CMAESAlgo {
    fn to_c_int(self) -> c_int {
        unsafe {
            match self {
                CMAESAlgo::Default => raw::const_CMAES_DEFAULT(),
                CMAESAlgo::IPOP => raw::const_IPOP_CMAES(),
                CMAESAlgo::BIPOP => raw::const_BIPOP_CMAES(),
                CMAESAlgo::ACMAES => raw::const_aCMAES(),
                CMAESAlgo::AIPOP => raw::const_aIPOP_CMAES(),
                CMAESAlgo::ABIPOP => raw::const_aBIPOP_CMAES(),
                CMAESAlgo::SEP => raw::const_sepCMAES(),
                CMAESAlgo::SEPIPOP => raw::const_sepIPOP_CMAES(),
                CMAESAlgo::SEPBIPOP => raw::const_sepBIPOP_CMAES(),
                CMAESAlgo::SEPA => raw::const_sepaCMAES(),
                CMAESAlgo::SEPAIPOP => raw::const_sepaIPOP_CMAES(),
                CMAESAlgo::SEPABIPOP => raw::const_sepaBIPOP_CMAES(),
                CMAESAlgo::VD => raw::const_VD_CMAES(),
                CMAESAlgo::VDIPOP => raw::const_VD_IPOP_CMAES(),
                CMAESAlgo::VDBIPOP => raw::const_VD_BIPOP_CMAES(),
            }
        }
    }
}

impl Default for CMAESAlgo {
    fn default() -> Self {
        CMAESAlgo::Default
    }
}

/// This structure keeps track of CMA-ES parameters.
///
/// The default() method has population size of 10, default algorithm, uses sigma of 1.0 and does
/// not use surrogates.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CMAESParameters {
    m_pop_size: usize,
    m_sigma: f64,
    m_algo: CMAESAlgo,
    m_use_surrogates: bool,
}

impl CMAESParameters {
    pub fn pop_size(&self) -> usize {
        self.m_pop_size
    }

    pub fn set_pop_size(&mut self, pop_size: usize) {
        self.m_pop_size = pop_size;
    }

    pub fn sigma(&self) -> f64 {
        self.m_sigma
    }

    pub fn set_sigma(&mut self, sigma: f64) {
        self.m_sigma = sigma;
    }

    pub fn algo(&self) -> CMAESAlgo {
        self.m_algo
    }

    pub fn set_algo(&mut self, algo: CMAESAlgo) {
        self.m_algo = algo;
    }

    pub fn use_surrogates(&self) -> bool {
        self.m_use_surrogates
    }

    pub fn set_use_surrogates(&mut self, surrogates: bool) {
        self.m_use_surrogates = surrogates;
    }
}

impl Default for CMAESParameters {
    fn default() -> Self {
        CMAESParameters {
            m_pop_size: 10,
            m_algo: CMAESAlgo::Default,
            m_sigma: 1.0,
            m_use_surrogates: false,
        }
    }
}

pub struct Userdata<'a> {
    evaluate: Box<dyn Fn(&'a [f64]) -> f64 + 'a>,
    iterate: Box<dyn FnMut() -> () + 'a>,
    d: usize,
}

extern "C" fn global_evaluate(
    candidate: *const c_double,
    _: *const c_int,
    userdata: *const c_void,
) -> f64 {
    unsafe {
        let userdata_ptr: *const Userdata = userdata as *const Userdata;
        let mut vec = Vec::with_capacity((*userdata_ptr).d);
        for i in 0..(*userdata_ptr).d {
            vec.push(*candidate.offset(i as isize));
        }
        let ev = &(*userdata_ptr).evaluate;
        ev(&vec)
    }
}

extern "C" fn global_iterator(userdata: *const c_void) {
    unsafe {
        let userdata_ptr: *mut Userdata = userdata as *mut Userdata;
        let it = &mut (*userdata_ptr).iterate;
        it();
    }
}

/// Optimizes some structure, minimizing the value from evaluate function.
///
/// Returns the candidate that managed to achieve lowest value from evaluate function.
///
/// ```
/// extern crate rcmaes;
///
/// use rcmaes::{optimize, CMAESParameters, Vectorizable};
///
/// // This example "trains" a very simple model to go to point (2, 8)
/// #[derive(Clone)]
/// struct TwoPoly {
///     x: f64,
///     y: f64,
/// }
///
/// impl Vectorizable for TwoPoly {
///     type Context = ();
///
///     fn to_vec(&self) -> (Vec<f64>, Self::Context) {
///         (vec![self.x, self.y], ())
///     }
///
///     fn from_vec(vec: &[f64], _: &Self::Context) -> Self {
///         TwoPoly {
///             x: vec[0],
///             y: vec[1],
///         }
///     }
/// }
///
/// fn train_model()
/// {
///     let optimized = optimize(
///         &TwoPoly { x: 5.0, y: 6.0 },
///         &CMAESParameters::default(),
///         |twopoly| (twopoly.x - 2.0).abs() + (twopoly.y - 8.0).abs(),
///     ).unwrap();
///
///     let model = optimized.0;
///     assert!((model.x - 2.0).abs() < 0.00001);
///     assert!((model.y - 8.0).abs() < 0.00001);
/// }
/// ```
#[inline]
pub fn optimize<T, F>(initial: &T, params: &CMAESParameters, evaluate: F) -> Option<(T, f64)>
where
    T: Vectorizable + Clone,
    F: Fn(T) -> f64,
{
    let it: Option<fn() -> ()> = None;
    optimize_raw(initial, params, evaluate, it)
}

/// Same as optimize() but also calls a function just before generating a new population.
///
/// This can be useful if you have some kind of randomized evaluation environment (e.g. simulation,
/// but you may want optimize_with_batch instead if your situation is like that))
/// and you want to evaluate each candidate solution on the same environment each round. You can
/// use the iterate function to generate a new environment
#[inline]
pub fn optimize_with_iterate<T, F, I>(
    initial: &T,
    params: &CMAESParameters,
    evaluate: F,
    iterator: I,
) -> Option<(T, f64)>
where
    T: Vectorizable + Clone,
    F: Fn(T) -> f64,
    I: FnMut(),
{
    optimize_raw(initial, params, evaluate, Some(iterator))
}

/// Same as optimize_with_iterate but eva
pub fn optimize_with_batch<T, F, I, B>(
    initial: &T,
    params: &CMAESParameters,
    evaluate: F,
    mut make_batch: I,
) -> Option<(T, f64)>
where
    T: Vectorizable + Clone,
    F: Fn(&B, T) -> f64,
    I: FnMut() -> B,
{
    let last_batch = RwLock::new(make_batch());

    optimize_raw(
        initial,
        params,
        |item| {
            let lb = last_batch.read().unwrap();
            evaluate(&*lb, item)
        },
        Some(|| {
            let mut lb = last_batch.write().unwrap();
            *lb = make_batch();
        }),
    )
}

fn optimize_raw<T, F, I>(
    initial: &T,
    params: &CMAESParameters,
    evaluate: F,
    iterator: Option<I>,
) -> Option<(T, f64)>
where
    T: Vectorizable + Clone,
    F: Fn(T) -> f64,
    I: FnMut(),
{
    let mut iterator = iterator;
    let best_so_far: Arc<Mutex<Option<(T, f64)>>> = Arc::new(Mutex::new(None));
    let (mut initial_vec, ctx) = initial.to_vec();

    let call = |thing_vec| {
        let score = evaluate(T::from_vec(thing_vec, &ctx));
        let mut bsf = best_so_far.lock().unwrap();
        match *bsf {
            None => *bsf = Some((T::from_vec(thing_vec, &ctx), score)),
            Some((_, old_score)) => {
                if old_score >= score {
                    *bsf = Some((T::from_vec(thing_vec, &ctx), score));
                }
            }
        }
        score
    };

    let it = || match iterator {
        None => (),
        Some(ref mut iterator_fun) => iterator_fun(),
    };

    let vec_ptr: *mut f64 = initial_vec.as_mut_ptr();

    let userdata = Userdata {
        d: initial_vec.len(),
        evaluate: Box::new(call),
        iterate: Box::new(it),
    };

    if initial_vec.len() == 0 {
        return None;
    }

    unsafe {
        raw::cmaes_optimize(
            if params.m_use_surrogates { 1 } else { 0 },
            params.algo().to_c_int(),
            vec_ptr,
            params.sigma(),
            params.pop_size() as i32,
            initial_vec.len() as u64,
            global_evaluate,
            global_iterator,
            (&userdata as *const Userdata) as *const c_void,
        );
    }
    let bsp = best_so_far.lock().unwrap();
    match *bsp {
        None => None,
        Some((ref thing, score)) => Some((thing.clone(), score)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TwoPoly {
        x: f64,
        y: f64,
    }

    impl Vectorizable for TwoPoly {
        type Context = ();

        fn to_vec(&self) -> (Vec<f64>, Self::Context) {
            (vec![self.x, self.y], ())
        }

        fn from_vec(vec: &[f64], _: &Self::Context) -> Self {
            TwoPoly {
                x: vec[0],
                y: vec[1],
            }
        }
    }

    #[test]
    pub fn test_2polynomial() {
        let optimized = optimize(
            &TwoPoly { x: 5.0, y: 6.0 },
            &CMAESParameters::default(),
            |twopoly| (twopoly.x - 2.0).abs() + (twopoly.y - 8.0).abs(),
        )
        .unwrap();
        assert!((optimized.0.x - 2.0).abs() < 0.00001);
        assert!((optimized.0.y - 8.0).abs() < 0.00001);
    }

    #[test]
    pub fn test_2polynomial_it() {
        let mut val: i64 = 0;
        let optimized = optimize_with_iterate(
            &TwoPoly { x: 5.0, y: 6.0 },
            &CMAESParameters::default(),
            |twopoly| (twopoly.x - 25.0).abs() + (twopoly.y - 1.0).abs(),
            || val += 1,
        )
        .unwrap();
        assert!((optimized.0.x - 25.0).abs() < 0.00001);
        assert!((optimized.0.y - 1.0).abs() < 0.00001);
        assert!(val > 0);
    }
}
