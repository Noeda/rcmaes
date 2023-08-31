/*
 * CoSyNe: A CoSyNe implementation in Rust
 *
 * https://people.idsia.ch/~juergen/gomez08a.pdf
 */

use crate::vectorizable::Vectorizable;
use rand::prelude::*;

#[derive(Clone, Debug)]
pub struct Cosyne<T: Vectorizable> {
    // NxM matrix, one column is one parameter.
    population_raw: Vec<f64>,
    dimension: usize,
    settings: CosyneSettings,
    ctx: T::Context,
}

#[derive(Clone, Debug)]
pub struct CosyneSettings {
    subpop_size: usize,
    num_pop_replacement: usize,
    sigma: f64,
    shrinkage_multiplier: f64,
    sampler: CosyneSampler,
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
                value + rng.gen_range(low_sigma..high_sigma)
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
}

impl<T: Clone + Vectorizable> Cosyne<T> {
    pub fn new(initial: &T, settings: &CosyneSettings) -> Self {
        assert!(settings.num_pop_replacement <= settings.subpop_size);

        let (vec, ctx) = initial.to_vec();
        let mut pop: Vec<f64> = Vec::with_capacity(settings.subpop_size * vec.len());
        let mut rng = thread_rng();
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

        let mut rng = thread_rng();
        candidates.sort_unstable_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
}
