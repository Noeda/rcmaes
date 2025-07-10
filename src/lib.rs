extern crate libc;
extern crate rand;
extern crate rand_distr;
extern crate rayon;

pub mod raw;

pub mod ball_descend;
pub mod cosyne;
pub mod mapelites;
#[cfg(feature = "pycmaes")]
pub mod pycmaes;
pub mod trending_down_test;
pub mod vectorizable;

pub use vectorizable::Vectorizable;

use libc::{c_double, c_int, c_void, size_t};
use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;

/// This structure contains some metadata about learning in last iteration of CMA-ES population.
///
/// Useful for displaying things.
#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct IterationReport {
    best_score: f64,
    worst_score: f64,
    average_score: f64,
}

impl IterationReport {
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    pub fn worst_score(&self) -> f64 {
        self.worst_score
    }

    pub fn average_score(&self) -> f64 {
        self.average_score
    }
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

// Pointer smuggler across thread boundary.
#[derive(Clone, Ord, Eq, PartialOrd, PartialEq)]
struct PointerSmuggler {
    ptr: *const c_void,
}

unsafe impl Send for PointerSmuggler {}
unsafe impl Sync for PointerSmuggler {}

impl From<PointerSmuggler> for *const c_void {
    fn from(ps: PointerSmuggler) -> Self {
        ps.ptr
    }
}

impl From<&PointerSmuggler> for *const c_void {
    fn from(ps: &PointerSmuggler) -> Self {
        ps.ptr
    }
}

impl From<*const c_void> for PointerSmuggler {
    fn from(ptr: *const c_void) -> Self {
        PointerSmuggler { ptr }
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

/// Different surrogate models.
#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub enum CMAESSurrogateModel {
    RSVMSurrogates(usize),
}

fn m_use_surrogates_to_c_int(surrogates: Option<CMAESSurrogateModel>) -> c_int {
    match surrogates {
        Some(CMAESSurrogateModel::RSVMSurrogates(_)) => 1,
        None => 0,
    }
}

fn m_use_surrogates_to_rsvm_surrogates(surrogates: Option<CMAESSurrogateModel>) -> c_int {
    match surrogates {
        Some(CMAESSurrogateModel::RSVMSurrogates(x)) => x as c_int,
        None => 0,
    }
}

/// Should we stop or continue training?
#[derive(Eq, Ord, PartialEq, PartialOrd, Debug, Clone, Copy)]
pub enum CMAESContinue {
    Stop,
    Continue,
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
    m_use_surrogates: Option<CMAESSurrogateModel>,
    m_use_elitism: bool,
    m_noisy: bool,
}

/// Returns a suggested population size by dimension.
pub fn suggested_pop_size_by_dimension(dimension: usize) -> usize {
    4 + (3.0 * (dimension as f64)).ln().floor() as usize
}

impl CMAESParameters {
    pub fn new() -> Self {
        CMAESParameters::default()
    }

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

    pub fn use_surrogates(&self) -> Option<CMAESSurrogateModel> {
        self.m_use_surrogates
    }

    pub fn set_use_surrogates(&mut self, surrogates: Option<CMAESSurrogateModel>) {
        self.m_use_surrogates = surrogates;
    }

    pub fn use_elitism(&self) -> bool {
        self.m_use_elitism
    }

    pub fn set_use_elitism(&mut self, elitism: bool) {
        self.m_use_elitism = elitism;
    }

    pub fn noisy(&self) -> bool {
        self.m_noisy
    }

    pub fn set_noisy(&mut self, noisy: bool) {
        self.m_noisy = noisy;
    }
}

impl Default for CMAESParameters {
    fn default() -> Self {
        CMAESParameters {
            m_pop_size: 10,
            m_algo: CMAESAlgo::Default,
            m_sigma: 1.0,
            m_use_surrogates: None,
            m_use_elitism: false,
            m_noisy: false,
        }
    }
}

struct Userdata<'a> {
    evaluate: Box<dyn Fn(&'a [f64]) -> (f64, CMAESContinue) + 'a>,
    iterate: Box<dyn FnMut() -> () + 'a>,
    d: usize,
}

struct MVars {
    outgoing_free: VecDeque<(usize, PointerSmuggler)>,
    outgoing_taken: BTreeMap<usize, PointerSmuggler>,
    incoming: BTreeMap<usize, PointerSmuggler>,
}

struct UserdataAskTell {
    mvars: Arc<Mutex<MVars>>,
    mvars_cond: Arc<Condvar>,
    mvars_ready_cond: Arc<(Mutex<bool>, Condvar)>,

    // populated by libcmaes library by calling our global_iterator_asktell
    sigma: Arc<Mutex<f64>>,
}

extern "C" fn global_evaluate(
    candidate: *const c_double,
    _: *const c_int,
    userdata: *const c_void,
    stop: *mut c_int,
) -> f64 {
    unsafe {
        let userdata_ptr: *const Userdata = userdata as *const Userdata;
        let mut vec = Vec::with_capacity((*userdata_ptr).d);
        for i in 0..(*userdata_ptr).d {
            vec.push(*candidate.offset(i as isize));
        }
        let ev = &(*userdata_ptr).evaluate;
        let (result, should_stop) = ev(&vec);
        if should_stop == CMAESContinue::Stop {
            *stop = 1;
        }
        result
    }
}

extern "C" fn global_iterator(userdata: *const c_void, _sigma: c_double) {
    unsafe {
        let userdata_ptr: *mut Userdata = userdata as *mut Userdata;
        let it = &mut (*userdata_ptr).iterate;
        it();
    }
}

extern "C" fn global_iterator_asktell(userdata: *const c_void, sigma: c_double) {
    unsafe {
        let userdata_ptr: *const UserdataAskTell = userdata as *const UserdataAskTell;
        let mut lock_sigma = (*userdata_ptr).sigma.lock().unwrap();
        *lock_sigma = sigma;
    }
}

extern "C" fn global_tell_mvars(
    userdata: *const c_void,
    mvars_outgoing: *const *const c_void,
    mvars_incoming: *const *const c_void,
    num_mvars: size_t,
) {
    unsafe {
        let userdata = userdata as *const UserdataAskTell;
        let userdata_ptr: *const UserdataAskTell = userdata;
        {
            let mut mvars = (*userdata_ptr).mvars.lock().unwrap();
            for i in 0..num_mvars {
                let mvar_outgoing_ptr = *mvars_outgoing.offset(i as isize);
                mvars.outgoing_free.push_back((i, mvar_outgoing_ptr.into()));
                let mvar_incoming_ptr = *mvars_incoming.offset(i as isize);
                mvars.incoming.insert(i, mvar_incoming_ptr.into());
            }
        }
        {
            let mut ready = (&(*userdata).mvars_ready_cond).0.lock().unwrap();
            *ready = true;
        }
        let _mrc = (&(*userdata).mvars_ready_cond).0.lock().unwrap();
        (&(*userdata).mvars_ready_cond).1.notify_all();
    }
}

extern "C" fn global_wait_until_dead(userdata: *const c_void) {
    unsafe {
        let userdata_ptr: *const UserdataAskTell = userdata as *const UserdataAskTell;
        {
            let mut mvars = (*userdata_ptr).mvars.lock().unwrap();
            mvars.outgoing_free.clear();
            mvars.outgoing_taken.clear();
            mvars.incoming.clear();
        }
    }
}

/// Optimizes some structure, minimizing the value from evaluate function.
///
/// Returns the candidate that managed to achieve lowest value from evaluate function.
///
/// ```
/// extern crate rcmaes;
///
/// use rcmaes::{optimize, CMAESParameters, Vectorizable, CMAESContinue};
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
///         |twopoly| ((twopoly.x - 2.0).abs() + (twopoly.y - 8.0).abs(), CMAESContinue::Continue),
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
    F: Fn(T) -> (f64, CMAESContinue),
{
    let it: Option<fn(IterationReport, T) -> ()> = None;
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
    F: Fn(T) -> (f64, CMAESContinue),
    I: FnMut(IterationReport, T),
{
    optimize_raw(initial, params, evaluate, Some(iterator))
}

/// Same as optimize_with_iterate but some batch stuff.
pub fn optimize_with_batch<T, F, I, B>(
    initial: &T,
    params: &CMAESParameters,
    evaluate: F,
    mut make_batch: I,
) -> Option<(T, f64)>
where
    T: Vectorizable + Clone,
    F: Fn(&B, T) -> (f64, CMAESContinue),
    I: FnMut(IterationReport, T) -> B,
{
    let last_batch = RwLock::new(make_batch(
        IterationReport {
            best_score: 0.0,
            worst_score: 0.0,
            average_score: 0.0,
        },
        initial.clone(),
    ));

    optimize_raw(
        initial,
        params,
        |item| {
            let lb = last_batch.read().unwrap();
            evaluate(&*lb, item)
        },
        Some(|iteration_report, model| {
            let mut lb = last_batch.write().unwrap();
            *lb = make_batch(iteration_report, model);
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
    F: Fn(T) -> (f64, CMAESContinue),
    I: FnMut(IterationReport, T),
{
    let mut iterator = iterator;
    let best_so_far: Arc<Mutex<Option<(T, f64)>>> = Arc::new(Mutex::new(None));
    let (mut initial_vec, ctx) = initial.to_vec();

    let best_score_seen_in_last_batch: Arc<Mutex<(f64, T)>> =
        Arc::new(Mutex::new((100000000000000000000.0, initial.clone())));
    let worst_score_seen_in_last_batch: Arc<Mutex<f64>> =
        Arc::new(Mutex::new(-100000000000000000000.0));
    let total_score_seen_in_last_batch: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));

    let call = |thing_vec| {
        let model = T::from_vec(thing_vec, &ctx);
        let (score, stop) = evaluate(model.clone());
        {
            let mut sc = best_score_seen_in_last_batch.lock().unwrap();
            if sc.0 > score {
                *sc = (score, model);
            }
        }
        {
            let mut sc = worst_score_seen_in_last_batch.lock().unwrap();
            if *sc < score {
                *sc = score;
            }
        }
        {
            let mut sc = total_score_seen_in_last_batch.lock().unwrap();
            *sc += score;
        }
        let mut bsf = best_so_far.lock().unwrap();
        match *bsf {
            None => *bsf = Some((T::from_vec(thing_vec, &ctx), score)),
            Some((_, old_score)) => {
                if old_score >= score {
                    *bsf = Some((T::from_vec(thing_vec, &ctx), score));
                }
            }
        }
        (score, stop)
    };

    let it = || {
        let average_score;
        {
            let mut sc = total_score_seen_in_last_batch.lock().unwrap();
            average_score = *sc / params.pop_size() as f64;
            *sc = 0.0;
        }
        let best_score;
        {
            let mut sc = best_score_seen_in_last_batch.lock().unwrap();
            best_score = sc.clone();
            *sc = (10000000000000000000.0, initial.clone());
        }
        let worst_score;
        {
            let mut sc = worst_score_seen_in_last_batch.lock().unwrap();
            worst_score = *sc;
            *sc = -10000000000000000000.0;
        }
        match iterator {
            None => (),
            Some(ref mut iterator_fun) => iterator_fun(
                IterationReport {
                    best_score: best_score.0,
                    worst_score,
                    average_score,
                },
                best_score.1,
            ),
        };
    };

    let vec_ptr: *mut f64 = initial_vec.as_mut_ptr();

    let userdata = Userdata {
        d: initial_vec.len(),
        evaluate: Box::new(call),
        iterate: Box::new(it),
    };

    if initial_vec.is_empty() {
        return None;
    }

    unsafe {
        raw::cmaes_optimize(
            if params.m_noisy { 1 } else { 0 },
            if params.m_use_elitism { 1 } else { 0 },
            m_use_surrogates_to_c_int(params.m_use_surrogates),
            params.algo().to_c_int(),
            vec_ptr,
            params.sigma(),
            params.pop_size() as i32,
            initial_vec.len() as u64,
            Some(global_evaluate),
            Some(global_iterator),
            None,
            None,
            m_use_surrogates_to_rsvm_surrogates(params.m_use_surrogates),
            (&userdata as *const Userdata) as *const c_void,
        );
    }
    let bsp = best_so_far.lock().unwrap();
    match *bsp {
        None => None,
        Some((ref thing, score)) => Some((thing.clone(), score)),
    }
}

pub struct CMAES<T: Vectorizable> {
    dim: usize,
    worker: Option<thread::JoinHandle<()>>,
    phantom: PhantomData<T>,
    userdata: Pin<Box<UserdataAskTell>>,
    pop_size: usize,
    asked: bool,
    dead: bool,
    guessed_openmp_nthreads: usize,
    ctx: <T as Vectorizable>::Context,
}

#[derive(Clone, Debug)]
pub struct CMAESCandidate<T> {
    item: T,
    score: f64,
    idx: usize,
}

impl<T> CMAESCandidate<T> {
    pub fn item(&self) -> &T {
        &self.item
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn set_score(&mut self, score: f64) {
        self.score = score;
    }

    pub fn idx(&self) -> usize {
        self.idx
    }
}

// Drop needed to override default 'worker' drop behavior that detaches a thread.
impl<T: Vectorizable> Drop for CMAES<T> {
    fn drop(&mut self) {
        {
            let mvars = self.userdata.mvars.lock().unwrap();
            for (_, i) in mvars.outgoing_free.iter() {
                unsafe {
                    raw::cmaes_mark_as_dead_mvar(i.into());
                }
            }
            for (_, i) in mvars.incoming.iter() {
                unsafe {
                    raw::cmaes_mark_as_dead_mvar(i.into());
                }
            }
            for (_, i) in mvars.outgoing_taken.iter() {
                unsafe {
                    raw::cmaes_mark_as_dead_mvar(i.into());
                }
            }
        }

        let mut empty = None;
        std::mem::swap(&mut self.worker, &mut empty);
        if let Some(worker) = empty {
            worker.join().unwrap();
        }
    }
}

impl<T: Clone + Vectorizable> CMAES<T> {
    pub fn new(initial: &T, params: &CMAESParameters) -> Self {
        let (mut vec, ctx) = initial.to_vec();
        let dim = vec.len();
        let params = params.clone();

        let userdata = Box::pin(UserdataAskTell {
            mvars: Arc::new(Mutex::new(MVars {
                outgoing_free: VecDeque::new(),
                outgoing_taken: BTreeMap::new(),
                incoming: BTreeMap::new(),
            })),
            mvars_cond: Arc::new(Condvar::new()),
            mvars_ready_cond: Arc::new((Mutex::new(false), Condvar::new())),
            sigma: Arc::new(Mutex::new(params.sigma())),
        });
        let userdata_ptr: PointerSmuggler =
            (userdata.deref() as *const UserdataAskTell as *const c_void).into();

        let worker = thread::spawn(move || {
            let userdata_ptr: *const c_void = userdata_ptr.into();
            let vec_ptr: *mut f64 = vec.as_mut_ptr();
            unsafe {
                raw::cmaes_optimize(
                    if params.m_noisy { 1 } else { 0 },
                    if params.m_use_elitism { 1 } else { 0 },
                    m_use_surrogates_to_c_int(params.m_use_surrogates),
                    params.algo().to_c_int(),
                    vec_ptr,
                    params.sigma(),
                    params.pop_size() as i32,
                    dim as u64,
                    None,
                    Some(global_iterator_asktell),
                    Some(global_tell_mvars),
                    Some(global_wait_until_dead),
                    m_use_surrogates_to_rsvm_surrogates(params.m_use_surrogates),
                    (userdata_ptr as *const UserdataAskTell) as *const c_void,
                )
            }
        });

        {
            loop {
                let mrc_lock = userdata.mvars_ready_cond.0.lock().unwrap();
                let mrc = userdata.mvars_ready_cond.1.wait(mrc_lock).unwrap();
                if !*mrc {
                    continue;
                }
                break;
            }
        }

        let guessed_openmp_nthreads = unsafe { raw::guess_number_of_omp_threads() };

        CMAES {
            dim,
            guessed_openmp_nthreads: guessed_openmp_nthreads as usize,
            pop_size: params.pop_size(),
            worker: Some(worker),
            phantom: PhantomData,
            userdata: userdata,
            asked: false,
            dead: false,
            ctx,
        }
    }

    pub fn ask(&mut self) -> Vec<CMAESCandidate<T>> {
        assert!(!self.asked);
        assert!(!self.dead);
        self.asked = true;

        self.guessed_openmp_nthreads = unsafe { raw::guess_number_of_omp_threads() as usize };

        let fifty_ms = 50000;

        let mut result = Vec::with_capacity(self.pop_size);
        {
            while result.len() < self.pop_size {
                let mut mvars = self.userdata.mvars.lock().unwrap();
                while mvars.outgoing_free.is_empty() {
                    mvars = self.userdata.mvars_cond.wait(mvars).unwrap();
                }
                let first_outgoing_mvar = mvars.outgoing_free.pop_front().unwrap();

                let mut success: c_int = 0;
                let success_mptr: *mut c_int = &mut success as *mut c_int;

                let result_item = unsafe {
                    if result.len() > 0 {
                        if result.len() < self.guessed_openmp_nthreads {
                            raw::cmaes_candidates_mvar_take_timeout(
                                first_outgoing_mvar.1.clone().into(),
                                fifty_ms,
                                success_mptr,
                            )
                        } else {
                            raw::cmaes_candidates_mvar_take_timeout(
                                first_outgoing_mvar.1.clone().into(),
                                0,
                                success_mptr,
                            )
                        }
                    } else {
                        raw::cmaes_candidates_mvar_take(
                            first_outgoing_mvar.1.clone().into(),
                            success_mptr,
                        )
                    }
                };

                if unsafe { *success_mptr } == 0 {
                    mvars.outgoing_free.push_front(first_outgoing_mvar);
                    return result;
                }
                mvars
                    .outgoing_taken
                    .insert(first_outgoing_mvar.0, first_outgoing_mvar.1);

                let params: *const f64 = result_item as *const f64;
                let vec: &[f64] = unsafe { std::slice::from_raw_parts(params, self.dim) };

                let item = T::from_vec(vec, &self.ctx);
                result.push(CMAESCandidate {
                    item,
                    score: 0.0,
                    idx: first_outgoing_mvar.0,
                });
            }
        }
        result
    }

    pub fn tell(&mut self, candidates: Vec<CMAESCandidate<T>>) {
        assert!(self.asked);
        assert!(!self.dead);
        self.asked = false;

        {
            let mut mvars = self.userdata.mvars.lock().unwrap();
            for candidate in candidates.iter() {
                let idx: usize = candidate.idx;

                let mvar_incoming = mvars.incoming.get(&idx).unwrap().clone();
                let ot = mvars.outgoing_taken.remove(&idx).unwrap();
                mvars.outgoing_free.push_back((idx, ot));

                let mut score: *mut c_void = std::ptr::null_mut();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &candidate.score as *const f64 as *const f64,
                        (&mut score) as *mut *mut c_void as *mut f64,
                        1,
                    );
                }

                let result =
                    unsafe { raw::cmaes_candidates_mvar_give(mvar_incoming.into(), score) };
                if result == 0 {
                    self.dead = true;
                    return;
                }
            }
        }
    }

    pub fn sigma(&self) -> f64 {
        let lock_sigma = self.userdata.sigma.lock().unwrap();
        *lock_sigma
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
            |twopoly| {
                (
                    (twopoly.x - 2.0).abs() + (twopoly.y - 8.0).abs(),
                    CMAESContinue::Continue,
                )
            },
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
            |twopoly| {
                (
                    (twopoly.x - 25.0).abs() + (twopoly.y - 1.0).abs(),
                    CMAESContinue::Continue,
                )
            },
            |_, _| val += 1,
        )
        .unwrap();
        assert!((optimized.0.x - 25.0).abs() < 0.00001);
        assert!((optimized.0.y - 1.0).abs() < 0.00001);
        assert!(val > 0);
    }

    #[test]
    pub fn test_stop_training() {
        let mut val: i64 = 0;
        let optimized = optimize_with_iterate(
            &TwoPoly { x: 5.0, y: 6.0 },
            &CMAESParameters::default(),
            |twopoly| {
                (
                    (twopoly.x - 25.0).abs() + (twopoly.y - 1.0).abs(),
                    CMAESContinue::Stop,
                )
            },
            |_, _| val += 1,
        )
        .unwrap();
        assert!((optimized.0.x - 25.0).abs() > 0.00001);
        assert!((optimized.0.y - 1.0).abs() > 0.00001);
    }

    #[test]
    pub fn ask_tell_default_params() {
        ask_tell_with_params(CMAESParameters::default());
    }

    #[test]
    pub fn ask_tell_surrogate_rsvm() {
        let mut params = CMAESParameters::default();
        params.set_use_surrogates(Some(CMAESSurrogateModel::RSVMSurrogates(50)));
        ask_tell_with_params(params);
    }

    fn ask_tell_with_params(params: CMAESParameters) {
        // run the whole thing many times (I had bugs that only happened rarely)
        for _idx in 0..10 {
            let mut cma = CMAES::new(&TwoPoly { x: 5.0, y: 6.0 }, &params);

            let mut epochs: usize = 1000;
            let mut best_score: f64 = 10000000.0;
            while epochs > 0 {
                epochs -= 1;
                let mut candidates = cma.ask();
                if candidates.is_empty() {
                    break;
                }
                for candidate in candidates.iter_mut() {
                    let score =
                        (candidate.item.x - 117.0).abs() + (candidate.item.y - (-87.0)).abs();
                    if score < best_score {
                        best_score = score;
                    }
                    candidate.set_score(score);
                }
                cma.tell(candidates);
            }
            assert!((best_score - 0.0).abs() < 0.00001);
        }
    }
}
