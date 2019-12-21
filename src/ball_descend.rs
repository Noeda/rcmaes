/// https://arxiv.org/abs/1911.06317
/// Gradientless Descent: High-Dimensional Zeroth-Order Optimization
/// Daniel Golovin, John Karro, Greg Kochanski, Chansoo Lee, Xingyou Song, Qiuyi Zhang
///
/// No idea if the implementation is entirely correct. Use at your own risk. It does empirically
/// work though.
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::vectorizable::Vectorizable;

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct BallDescendParameters {
    sigma_maximum: f64,
    sigma_minimum: f64,
    epochs: Option<u64>,
    report_to_stdout: bool,

    condition_number_bound: f64,
    use_fast_variant: bool,
}

impl Default for BallDescendParameters {
    fn default() -> Self {
        BallDescendParameters {
            sigma_maximum: 10.0,
            sigma_minimum: 0.0001,
            condition_number_bound: 10.0,
            report_to_stdout: false,
            epochs: None,
            use_fast_variant: true,
        }
    }
}

impl BallDescendParameters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sigma_maximum(&self) -> f64 {
        self.sigma_maximum
    }

    pub fn set_sigma_maximum(&mut self, sigma_maximum: f64) {
        self.sigma_maximum = sigma_maximum;
    }

    pub fn sigma_minimum(&self) -> f64 {
        self.sigma_minimum
    }

    pub fn set_sigma_minimum(&mut self, sigma_minimum: f64) {
        self.sigma_minimum = sigma_minimum;
    }

    pub fn report_to_stdout(&self) -> bool {
        self.report_to_stdout
    }

    pub fn set_report_to_stdout(&mut self, report: bool) {
        self.report_to_stdout = report;
    }

    pub fn epochs(&self) -> Option<u64> {
        self.epochs
    }

    pub fn set_epochs(&mut self, epochs: Option<u64>) {
        self.epochs = epochs;
    }

    pub fn use_fast_variant(&self) -> bool {
        self.use_fast_variant
    }

    pub fn set_use_fast_variant(&mut self, use_fast_variant: bool) {
        self.use_fast_variant = use_fast_variant;
    }

    pub fn condition_number_bound(&self) -> f64 {
        self.condition_number_bound
    }

    pub fn set_condition_number_bound(&mut self, condition_number_bound: f64) {
        self.condition_number_bound = condition_number_bound;
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
    F: Fn(&B, &T) -> f64 + Sync,
    I: FnMut(T, f64) -> B,
    B: Send + Sync,
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
    let mut last_iteration_score = None;
    let mut sigma_maximum = params.sigma_maximum();
    let sigma_minimum = params.sigma_minimum();

    let mut epochs: u64 = 0;

    let (vec, _) = initial.to_vec();

    #[allow(non_snake_case)]
    let H: u64 = ((vec.len() as f64)
        * params.condition_number_bound()
        * params.condition_number_bound().ln().ceil()) as u64;
    std::mem::drop(vec);

    loop {
        if params.use_fast_variant() && (epochs + 1) % H == 0 {
            sigma_maximum *= 0.5;
        }
        let test_item = make_batch(
            last_iteration_best.clone(),
            last_iteration_score.unwrap_or(0.0),
        );
        last_iteration_score = Some(evaluate(&test_item, &last_iteration_best));

        #[allow(non_snake_case)]
        let K = if params.use_fast_variant() {
            ((4.0 as f64) * params.condition_number_bound().sqrt())
                .ln()
                .ceil() as i64
        } else {
            (sigma_maximum / sigma_minimum).ln().ceil() as i64
        };

        let krange: Vec<i64> = if params.use_fast_variant() {
            (-K..=K).collect()
        } else {
            (0..=K).collect()
        };

        let evaluations: Vec<(T, f64)> = krange
            .par_iter()
            .filter_map(|k| {
                let s = (2.0 as f64).powf(-((*k) as f64)) * sigma_maximum;
                let perturbed = make_perturbed_candidate(&last_iteration_best, s);
                let perturbed_score = evaluate(&test_item, &perturbed);
                if Some(perturbed_score) < last_iteration_score || last_iteration_score.is_none() {
                    Some((perturbed, perturbed_score))
                } else {
                    None
                }
            })
            .collect();

        for (candidate_model, candidate_score) in evaluations.into_iter() {
            if last_iteration_score.is_none() || Some(candidate_score) < last_iteration_score {
                last_iteration_best = candidate_model;
                last_iteration_score = Some(candidate_score);
            }
        }

        if params.report_to_stdout() {
            if params.use_fast_variant() {
                println!(
                    "Current score: {:?} Epoch {} H {} K {} sigma_maximum {}",
                    last_iteration_score, epochs, H, K, sigma_maximum
                );
            } else {
                println!(
                    "Current score: {:?} Epoch {} K {}",
                    last_iteration_score, epochs, K
                );
            }
        }
        epochs += 1;
        if params.epochs().is_some() && Some(epochs) >= params.epochs() {
            break;
        }
    }

    last_iteration_best
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
        params.set_epochs(Some(1000));
        let optimized = optimize_with_batch(
            &TwoPoly { x: 5.0, y: 6.0 },
            params,
            |_, twopoly| (twopoly.x - 2.0).abs() + (twopoly.y - 8.0).abs(),
            |_, _| vec![0 as usize].into_iter(),
        );
        assert!((optimized.x - 2.0).abs() < 0.001);
        assert!((optimized.y - 8.0).abs() < 0.001);
    }

    #[test]
    pub fn test_2polynomial_fast_variant() {
        let mut params = BallDescendParameters::default();
        params.set_epochs(Some(1000));
        params.set_use_fast_variant(true);
        let optimized = optimize_with_batch(
            &TwoPoly { x: 5.0, y: 6.0 },
            params,
            |_, twopoly| (twopoly.x - 2.0).abs() + (twopoly.y - 8.0).abs(),
            |_, _| vec![0 as usize].into_iter(),
        );
        assert!((optimized.x - 2.0).abs() < 0.001);
        assert!((optimized.y - 8.0).abs() < 0.001);
    }
}
