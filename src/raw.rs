// This file contains the raw stuff to libcmaes
use libc::{c_double, c_int, c_void, size_t};

#[link(name = "rcmaesglue", kind = "static")]
#[link(name = "cmaes")]
extern "C" {
    pub fn cmaes_optimize(
        noisy: c_int,
        use_elitism: c_int,
        use_surrogates: c_int,
        algo: c_int,
        initial: *mut c_double,
        sigma: c_double,
        lambda: c_int,
        num_coords: u64,
        evaluate: Option<
            extern "C" fn(*const c_double, *const c_int, *const c_void, *mut c_int) -> c_double,
        >,
        iterator: Option<extern "C" fn(*const c_void, c_double) -> ()>,
        tell_mvars: Option<
            extern "C" fn(*const c_void, *const *const c_void, *const *const c_void, size_t) -> (),
        >,
        wait_until_dead: Option<extern "C" fn(*const c_void) -> ()>,
        userdata: *const c_void,
    );

    pub fn cmaes_candidates_mvar_take(mvar: *const c_void, success: *mut c_int) -> *const c_void;
    pub fn cmaes_candidates_mvar_take_timeout(
        mvar: *const c_void,
        microseconds: i64,
        success: *mut c_int,
    ) -> *const c_void;
    pub fn cmaes_candidates_mvar_give(mvar: *const c_void, content: *const c_void) -> c_int;
    pub fn cmaes_candidates_mvar_num_waiters(mvar: *const c_void) -> size_t;
    pub fn cmaes_mark_as_dead_mvar(mvar: *const c_void);

    pub fn guess_number_of_omp_threads() -> c_int;

    pub fn const_CMAES_DEFAULT() -> c_int;
    pub fn const_IPOP_CMAES() -> c_int;
    pub fn const_BIPOP_CMAES() -> c_int;
    pub fn const_aCMAES() -> c_int;
    pub fn const_aIPOP_CMAES() -> c_int;
    pub fn const_aBIPOP_CMAES() -> c_int;
    pub fn const_sepCMAES() -> c_int;
    pub fn const_sepIPOP_CMAES() -> c_int;
    pub fn const_sepBIPOP_CMAES() -> c_int;
    pub fn const_sepaCMAES() -> c_int;
    pub fn const_sepaIPOP_CMAES() -> c_int;
    pub fn const_sepaBIPOP_CMAES() -> c_int;
    pub fn const_VD_CMAES() -> c_int;
    pub fn const_VD_IPOP_CMAES() -> c_int;
    pub fn const_VD_BIPOP_CMAES() -> c_int;
}
