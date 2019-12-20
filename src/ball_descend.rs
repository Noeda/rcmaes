use rand::prelude::SliceRandom;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::sync::atomic::*;

use crate::vectorizable::Vectorizable;

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct BallDescendParameters {
    sigma: f64,
    trials_per_sigma: usize,
    diff_cutoff: f64,
    report_to_stdout: bool,
}

impl Default for BallDescendParameters {
    fn default() -> Self {
        BallDescendParameters {
            sigma: 1.0,
            trials_per_sigma: 8,
            diff_cutoff: 32.0,
            report_to_stdout: false,
        }
    }
}

impl BallDescendParameters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn set_sigma(&mut self, sigma: f64) {
        self.sigma = sigma;
    }

    pub fn trials_per_sigma(&self) -> usize {
        self.trials_per_sigma
    }

    pub fn set_trials_per_sigma(&mut self, trials: usize) {
        self.trials_per_sigma = trials;
    }

    pub fn diff_cutoff(&self) -> f64 {
        self.diff_cutoff
    }

    pub fn set_diff_cutoff(&mut self, diff_cutoff: f64) {
        self.diff_cutoff = diff_cutoff;
    }

    pub fn report_to_stdout(&self) -> bool {
        self.report_to_stdout
    }

    pub fn set_report_to_stdout(&mut self, report: bool) {
        self.report_to_stdout = report;
    }
}

pub fn optimize_with_batch<T, F, I, B>(
    initial: &T,
    params: BallDescendParameters,
    evaluate: F,
    mut make_batch: I,
) -> T
where
    T: Vectorizable + Clone + Sync + Send,
    F: Fn(B::Item, &T) -> f64 + Sync,
    I: FnMut(T) -> B,
    B: Iterator + rayon::iter::ParallelBridge,
    B::Item: Send + Sync + Clone,
    rayon::iter::IterBridge<B>: rayon::iter::ParallelIterator<Item = B::Item>,
{
    let make_perturbed_candidate = |m: &T, sigma: f64| -> T {
        let (mut t_as_vec, t_ctx) = m.to_vec();
        let mut rng = rand::thread_rng();
        let distr = Normal::new(0.0, sigma).unwrap();

        for v in t_as_vec.iter_mut() {
            *v += distr.sample(&mut rng);
        }

        T::from_vec(&t_as_vec, &t_ctx)
    };

    let mut last_iteration_best = initial.clone();
    let mut sigma = params.sigma();

    loop {
        let batch = make_batch(last_iteration_best.clone());

        let basics = AtomicUsize::new(0);
        let ups = AtomicUsize::new(0);
        let downs = AtomicUsize::new(0);

        let mut perturbations: Vec<Vec<f64>> = batch
            .par_bridge()
            .filter_map(|item| {
                let initial_score = evaluate(item.clone(), &last_iteration_best);

                let mut candidate = None;
                let mut diff = 1.0;
                loop {
                    let mut tries: usize = params.trials_per_sigma();

                    while tries > 0 {
                        let perturbed_up =
                            make_perturbed_candidate(&last_iteration_best, sigma * diff);
                        let perturbed_score_up = evaluate(item.clone(), &perturbed_up);
                        if perturbed_score_up < initial_score {
                            candidate = Some(perturbed_up);
                            if diff != 1.0 {
                                ups.fetch_add(1, Ordering::Relaxed);
                            } else {
                                basics.fetch_add(1, Ordering::Relaxed);
                            }
                            break;
                        }
                        let perturbed_down =
                            make_perturbed_candidate(&last_iteration_best, sigma * (1.0 / diff));
                        let perturbed_score_down = evaluate(item.clone(), &perturbed_down);
                        if perturbed_score_down < initial_score {
                            candidate = Some(perturbed_down);
                            if diff != 1.0 {
                                downs.fetch_add(1, Ordering::Relaxed);
                            } else {
                                basics.fetch_add(1, Ordering::Relaxed);
                            }
                            break;
                        }
                        tries -= 1;
                    }

                    if candidate.is_none() {
                        diff *= 2.0;
                        if diff >= params.diff_cutoff() {
                            return None;
                        }
                        continue;
                    }
                    break;
                }

                Some(quantify_perturbation(
                    &last_iteration_best,
                    &candidate.unwrap(),
                ))
            })
            .collect();

        let num_perturbations = perturbations.len();
        if num_perturbations == 0 {
            return last_iteration_best;
        }

        // Since we are summing a whole lot of floating point numbers together, we should do some
        // mitigation over biasing for items that appear first in the list.
        //
        // Floating point values start decreasing in precision once you start summing small numbers
        // to big numbers. We still have a naive summing algorithm but we'll at least shuffle the
        // things we are summing over.
        perturbations.shuffle(&mut rand::thread_rng());

        let mut averaged_perturbations = vec![0.0; perturbations[0].len()];
        for vec in perturbations.into_iter() {
            assert_eq!(vec.len(), averaged_perturbations.len());
            for idx in 0..vec.len() {
                averaged_perturbations[idx] += vec[idx];
            }
        }
        for x in averaged_perturbations.iter_mut() {
            *x /= num_perturbations as f64;
        }

        let (mut last_iteration_vec, ctx) = last_iteration_best.to_vec();
        assert_eq!(last_iteration_vec.len(), averaged_perturbations.len());
        for idx in 0..averaged_perturbations.len() {
            last_iteration_vec[idx] += averaged_perturbations[idx];
        }
        last_iteration_best = T::from_vec(&last_iteration_vec, &ctx);

        // If 75% of best iterations were high sigma/low sigma, then adjust it
        let ups = ups.load(Ordering::Relaxed);
        let downs = downs.load(Ordering::Relaxed);
        let basics = basics.load(Ordering::Relaxed);

        let total = ups + downs + basics;
        if ups > total * 3 / 4 {
            sigma *= 2.0;
        } else if downs > total * 3 / 4 {
            sigma *= 0.5;
        }

        if params.report_to_stdout() {
            println!(
                "Sigma: {}, ups {} basics {} downs {}",
                sigma, ups, basics, downs
            );
        }
    }
}

fn quantify_perturbation<T: Vectorizable>(first: &T, second: &T) -> Vec<f64> {
    let (vec1, _) = first.to_vec();
    let (vec2, _) = second.to_vec();

    assert_eq!(vec1.len(), vec2.len());

    let mut vec3 = Vec::with_capacity(vec1.len());

    for idx in 0..vec1.len() {
        vec3.push(vec2[idx] - vec1[idx]);
    }
    vec3
}

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
        let mut params = BallDescendParameters::default();
        params.set_trials_per_sigma(20);
        params.set_diff_cutoff(256.0);
        let optimized = optimize_with_batch(
            &TwoPoly { x: 5.0, y: 6.0 },
            params,
            |_, twopoly| (twopoly.x - 2.0).abs() + (twopoly.y - 8.0).abs(),
            |_| vec![0 as usize].into_iter(),
        );
        assert!((optimized.x - 2.0).abs() < 0.00001);
        assert!((optimized.y - 8.0).abs() < 0.00001);
    }
}
