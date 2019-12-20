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
