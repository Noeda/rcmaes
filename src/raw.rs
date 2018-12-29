// This file contains the raw stuff to libcmaes
use libc::{c_double, c_int, c_void};

#[link(name = "rcmaesglue", kind = "static")]
#[link(name = "cmaes")]
extern "C" {
    pub fn cmaes_optimize(
        use_surrogates: c_int,
        algo: c_int,
        initial: *mut c_double,
        sigma: c_double,
        lambda: c_int,
        num_coords: u64,
        evaluate: extern "C" fn(*const c_double, *const c_int, *const c_void) -> c_double,
        iterator: extern "C" fn(*const c_void) -> (),
        userdata: *const c_void,
    );

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
