/*
 * CoSyNe: A CoSyNe implementation in Rust
 *
 * https://people.idsia.ch/~juergen/gomez08a.pdf
 */

use crate::trending_down_test::series_is_trending_down_log;
use crate::vectorizable::Vectorizable;
use rand::prelude::*;
use rand::rng;

#[derive(Clone, Debug)]
pub struct Cosyne<T: Vectorizable> {
    // NxM matrix, one column is one parameter.
    population_raw: Vec<f64>,
    dimension: usize,
    settings: CosyneSettings,
    ctx: T::Context,

    // One epoch = one round of ask/tell
    epoch: usize,
    // If adaptive sigma is used, then this tracks the fitness over past
    // N epochs.
    //
    // Used in watermark-like way; we lit to grow to epoch_window*2, but
    // we don't call the statistical test except for epoch_window.
    //
    // TODO: adjust trending_down_test to be able to take VecDeque or a closure or something
    // (because inflexible interface is why this isn't a VecDeque).
    fitness_history: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct CosyneSettings {
    subpop_size: usize,
    num_pop_replacement: usize,
    sigma: f64,
    shrinkage_multiplier: f64,
    sampler: CosyneSampler,

    adapt_sigma: Option<AdaptSigmaSettings>,
}

#[derive(Clone, Debug)]
pub struct AdaptSigmaSettings {
    epoch_window: usize,
    // these are probabilities in log
    increase_sigma_threshold: f64,
    decrease_sigma_threshold: f64,
    sigma_increase: f64,
    sigma_decrease: f64,
    fitness_mean: FitnessMean,
}

#[derive(Clone, Debug, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub enum FitnessMean {
    Average,
    Median,
}

impl AdaptSigmaSettings {
    // epoch_window: How many epochs to consider for calculations
    // that try to figure out if fitness is improving or not.
    //
    // This also sets the minimum possible number of epochs before
    // a sigma adjustment.
    //
    // A good value is likely something like 50 at minimum; you
    // need enough that a statistical test on previous epochs yields
    // a meaningful result. 100 has been used successfully. Higher values
    // mean slower adapting sigma, but also that it is less likely to adapt
    // into the wrong direction because of a failed test.
    pub fn self_adapting_sigma(epoch_window: usize) -> Self {
        assert!(epoch_window > 0, "epoch_window must be greater than 0");
        AdaptSigmaSettings {
            epoch_window,
            increase_sigma_threshold: (0.99_f64).ln(),
            decrease_sigma_threshold: (0.5_f64).ln(),
            sigma_increase: 1.25,
            sigma_decrease: 0.5,
            fitness_mean: FitnessMean::Average,
        }
    }

    pub fn sigma_increase(mut self, sigma_increase: f64) -> Self {
        self.sigma_increase = sigma_increase;
        self
    }

    pub fn sigma_decrease(mut self, sigma_decrease: f64) -> Self {
        self.sigma_decrease = sigma_decrease;
        self
    }

    pub fn increase_sigma_threshold(mut self, threshold: f64) -> Self {
        assert!(
            threshold >= 0.0 && threshold <= 1.0,
            "increase_sigma_threshold must be in range [0.0..1.0]"
        );
        self.increase_sigma_threshold = threshold;
        self
    }

    pub fn decrease_sigma_threshold(mut self, threshold: f64) -> Self {
        assert!(
            threshold >= 0.0 && threshold <= 1.0,
            "decrease_sigma_threshold must be in range [0.0..1.0]"
        );
        self.decrease_sigma_threshold = threshold;
        self
    }

    pub fn fitness_mean(mut self, fitness_mean: FitnessMean) -> Self {
        self.fitness_mean = fitness_mean;
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CosyneSampler {
    Uniform,
    Gaussian,
    Cauchy,
}

impl CosyneSampler {
    pub fn sample<R: Rng>(
        &self,
        value: f64,
        sigma: f64,
        shrinkage_multiplier: f64,
        rng: &mut R,
    ) -> f64 {
        match self {
            CosyneSampler::Uniform => {
                let mut low_sigma = -sigma;
                let mut high_sigma = sigma;
                if value < 0.0 {
                    low_sigma *= shrinkage_multiplier;
                } else {
                    high_sigma *= shrinkage_multiplier;
                }
                value + rng.random_range(low_sigma..high_sigma)
            }
            CosyneSampler::Gaussian => {
                let dist = rand_distr::Normal::new(0.0, sigma).unwrap();
                value + dist.sample(rng) * shrinkage_multiplier
            }
            CosyneSampler::Cauchy => {
                let dist = rand_distr::Cauchy::new(0.0, sigma).unwrap();
                value + dist.sample(rng) * shrinkage_multiplier
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct CosyneCandidate<T> {
    item: T,
    score: f64,
    idx: usize,
}

impl<T: Clone> CosyneCandidate<T> {
    pub fn item(&self) -> &T {
        &self.item
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn set_score(&mut self, score: f64) {
        self.score = score;
    }
}

impl CosyneSettings {
    pub fn default() -> Self {
        CosyneSettings {
            subpop_size: 16,
            num_pop_replacement: 10,
            sigma: 1.0,
            shrinkage_multiplier: 1.0,
            sampler: CosyneSampler::Cauchy,
            adapt_sigma: Some(AdaptSigmaSettings::self_adapting_sigma(100)),
        }
    }

    // shrinkage multiplier is a type of regularization
    //
    // If the current value of some parameter is X, then, when generating a new mutation of X, the
    // new value is:
    //
    //  if X >= 0:   random_range [-X..X * shrinkage_multiplier]
    //  if X < 0:    random_range [-X * shrinkage_multiplier..X]
    //
    // Typical values might be something like 0.9 or 0.95, depending on problem. Using values
    // greater than 1 might cause explosion of parameter values. Using too low value may make it
    // hard for the model to learn anything as there's very strong tendency towards 0.
    //
    // Using multiplier of 1 (which is the default) effectively disables this feature.
    pub fn shrinkage_multiplier(mut self, multiplier: f64) -> Self {
        self.shrinkage_multiplier = multiplier;
        self
    }

    pub fn subpop_size(mut self, subpop_size: usize) -> Self {
        if subpop_size == 0 {
            panic!("subpop_size must be greater than 0");
        }
        self.subpop_size = subpop_size;
        self
    }

    pub fn num_pop_replacement(mut self, num_pop_replacement: usize) -> Self {
        if num_pop_replacement == 0 {
            panic!("num_pop_replacement must be greater than 0");
        }
        self.num_pop_replacement = num_pop_replacement;
        self
    }

    pub fn sigma(mut self, sigma: f64) -> Self {
        if sigma <= 0.0 {
            panic!("sigma must be greater than 0");
        }
        self.sigma = sigma;
        self
    }

    pub fn sampler(mut self, sampler: CosyneSampler) -> Self {
        self.sampler = sampler;
        self
    }

    // See AdaptSigmaSettings
    pub fn adapt_sigma(mut self, adapt_sigma: Option<AdaptSigmaSettings>) -> Self {
        self.adapt_sigma = adapt_sigma.clone();
        self
    }
}

impl<T: Clone + Vectorizable> Cosyne<T> {
    pub fn new(initial: &T, settings: &CosyneSettings) -> Self {
        assert!(settings.num_pop_replacement <= settings.subpop_size);

        let (vec, ctx) = initial.to_vec();
        let mut pop: Vec<f64> = Vec::with_capacity(settings.subpop_size * vec.len());
        let mut rng = rng();
        for idx in 0..settings.subpop_size * vec.len() {
            let individual_idx = idx / settings.subpop_size;
            let subpop_idx = idx % settings.subpop_size;
            if subpop_idx == 0 {
                pop.push(vec[individual_idx]);
            } else {
                let sample = settings.sampler.sample(
                    0.0,
                    settings.sigma,
                    settings.shrinkage_multiplier,
                    &mut rng,
                );
                pop.push(sample + vec[individual_idx]);
            }
        }

        Self {
            settings: settings.clone(),
            dimension: vec.len(),
            population_raw: pop,
            ctx,
            epoch: 0,
            fitness_history: vec![],
        }
    }

    pub fn settings(&self) -> CosyneSettings {
        self.settings.clone()
    }

    pub fn set_sigma(&mut self, sigma: f64) {
        self.settings.sigma = sigma;
    }

    pub fn sigma(&self) -> f64 {
        self.settings.sigma
    }

    pub fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn ask(&mut self) -> Vec<CosyneCandidate<T>> {
        let mut candidates: Vec<CosyneCandidate<T>> = Vec::with_capacity(self.settings.subpop_size);
        let mut candidate_vec: Vec<f64> = Vec::with_capacity(self.dimension);
        for idx in 0..self.settings.subpop_size {
            candidate_vec.truncate(0);
            for idx2 in 0..self.dimension {
                let offset = idx + idx2 * self.settings.subpop_size;
                candidate_vec.push(self.population_raw[offset]);
            }
            let item = T::from_vec(&candidate_vec, &self.ctx);
            candidates.push(CosyneCandidate {
                item,
                score: 0.0,
                idx: idx,
            });
        }
        candidates
    }

    pub fn tell(&mut self, mut candidates: Vec<CosyneCandidate<T>>) {
        assert_eq!(candidates.len(), self.settings.subpop_size);

        let mut rng = rng();
        candidates.sort_unstable_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // median calculated for adaptive sigma
        let median: f64 = if candidates.len() > 0 && candidates.len() % 2 == 0 {
            let idx1 = candidates.len() / 2 - 1;
            let idx2 = candidates.len() / 2;
            (candidates[idx1].score + candidates[idx2].score) / 2.0
        } else if candidates.len() > 0 {
            candidates[candidates.len() / 2].score
        } else {
            0.0
        };
        let avg: f64 = candidates.iter().map(|c| c.score).sum::<f64>() / candidates.len() as f64;

        for idx in 0..self.settings.num_pop_replacement {
            let (cand_vec, _) = candidates[idx].item.to_vec();
            let ridx = idx + (self.settings.subpop_size - self.settings.num_pop_replacement);
            for idx2 in 0..self.dimension {
                let old_value = cand_vec[idx2];
                let sample = self.settings.sampler.sample(
                    old_value,
                    self.settings.sigma,
                    self.settings.shrinkage_multiplier,
                    &mut rng,
                );
                let offset = candidates[ridx].idx + idx2 * self.settings.subpop_size;
                self.population_raw[offset] = sample;
            }
        }

        for idx in 0..self.dimension {
            let offset_start = idx * self.settings.subpop_size;
            let offset_end = (idx + 1) * self.settings.subpop_size;
            // Shuffle
            self.population_raw[offset_start..offset_end].shuffle(&mut rng);
        }

        self.epoch += 1;
        if let Some(ref adapt_sigma) = self.settings.adapt_sigma {
            match adapt_sigma.fitness_mean {
                FitnessMean::Median => self.fitness_history.push(median),
                FitnessMean::Average => self.fitness_history.push(avg),
            }
            let mut fitness_history_len = self.fitness_history.len();
            if fitness_history_len > adapt_sigma.epoch_window * 2 {
                self.fitness_history = (&self.fitness_history
                    [fitness_history_len - adapt_sigma.epoch_window..])
                    .to_owned();
            }
            fitness_history_len = self.fitness_history.len();
            if fitness_history_len >= adapt_sigma.epoch_window {
                // trending down (log) probability
                let n_up = adapt_sigma.increase_sigma_threshold;
                let n_down = adapt_sigma.decrease_sigma_threshold;
                let p = series_is_trending_down_log(
                    &self.fitness_history[fitness_history_len - adapt_sigma.epoch_window..],
                );
                if p > n_up {
                    self.settings.sigma *= adapt_sigma.sigma_increase;
                    self.fitness_history.clear();
                } else if p < n_down {
                    self.settings.sigma *= adapt_sigma.sigma_decrease;
                    self.fitness_history.clear();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
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
        let mut cosyne = Cosyne::new(&TwoPoly { x: 5.0, y: 6.0 }, &CosyneSettings::default());
        let mut best_seen = 100000.0;
        let mut optimized = TwoPoly { x: 100.0, y: 100.0 };
        for _ in 0..10000 {
            let mut cands = cosyne.ask();
            for cand in cands.iter_mut() {
                let item = cand.item();
                let score = (item.x - 2.0).abs() + (item.y - 8.0).abs();
                if score < best_seen {
                    best_seen = score;
                    optimized = item.clone();
                }
                cand.set_score(score);
            }
            cosyne.tell(cands);
        }
        assert!((optimized.x - 2.0).abs() < 0.01);
        assert!((optimized.y - 8.0).abs() < 0.01);
    }

    #[test]
    pub fn adaptable_sigma() {
        // We test by using a badly initialized sigma on purpose. The adapting sigma should be much
        // closer to the optimal value.

        let mut cosyne_adapting = Cosyne::new(
            &TwoPoly { x: 0.0, y: 0.0 },
            &CosyneSettings::default()
                .sigma(0.00001)
                .adapt_sigma(Some(AdaptSigmaSettings::self_adapting_sigma(50))),
        );
        let mut cosyne_nonadapting = Cosyne::new(
            &TwoPoly { x: 0.0, y: 0.0 },
            &CosyneSettings::default().sigma(0.00001).adapt_sigma(None),
        );

        let epochs = 2000;
        let mut best_adapting_score = std::f64::INFINITY;
        let mut best_nonadapting_score = std::f64::INFINITY;
        for _ in 0..epochs {
            let mut cands_adapting = cosyne_adapting.ask();
            let mut cands_nonadapting = cosyne_nonadapting.ask();

            for cand in cands_adapting.iter_mut() {
                let item = cand.item();
                // Target 10000.0 10000.0 for both x and y
                let score = (item.x - 10000.0).abs() + (item.y - 10000.0).abs();
                cand.set_score(score);
                if score < best_adapting_score {
                    best_adapting_score = score;
                }
            }
            for cand in cands_nonadapting.iter_mut() {
                let item = cand.item();
                let score = (item.x - 10000.0).abs() + (item.y - 10000.0).abs();
                cand.set_score(score);
                if score < best_nonadapting_score {
                    best_nonadapting_score = score;
                }
            }
            cosyne_adapting.tell(cands_adapting);
            cosyne_nonadapting.tell(cands_nonadapting);
        }
        assert!(best_adapting_score < best_nonadapting_score);
    }
}
