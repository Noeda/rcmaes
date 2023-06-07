/*
 * CoSyNe: A CoSyNe implementation in Rust
 *
 * https://people.idsia.ch/~juergen/gomez08a.pdf
 */

use crate::vectorizable::Vectorizable;
use rand::prelude::*;

#[derive(Clone, Debug)]
pub struct Cosyne<T: Vectorizable> {
    population: Vec<SubPopulation>,
    settings: CosyneSettings,
    ctx: T::Context,
}

#[derive(Clone, Debug)]
pub struct SubPopulation {
    individuals: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct CosyneSettings {
    subpop_size: usize,
    num_pop_replacement: usize,
    sigma: f64,
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
        }
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
}

impl<T: Clone + Vectorizable> Cosyne<T> {
    pub fn new(initial: &T, settings: &CosyneSettings) -> Self {
        assert!(settings.num_pop_replacement <= settings.subpop_size);

        let (vec, ctx) = initial.to_vec();
        let mut pop: Vec<SubPopulation> = vec![
            SubPopulation {
                individuals: Vec::with_capacity(settings.subpop_size)
            };
            vec.len()
        ];
        let mut rng = thread_rng();
        for (idx, v) in vec.iter().enumerate() {
            for idx2 in 0..settings.subpop_size {
                if idx2 == 0 {
                    pop[idx].individuals.push(v.clone());
                } else {
                    pop[idx]
                        .individuals
                        .push(v.clone() + rng.gen_range(-1.0..1.0));
                }
            }
        }

        Self {
            settings: settings.clone(),
            population: pop,
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
        let mut candidate_vec: Vec<f64> = Vec::with_capacity(self.population.len());
        for idx in 0..self.settings.subpop_size {
            candidate_vec.truncate(0);
            for idx2 in 0..self.population.len() {
                candidate_vec.push(self.population[idx2].individuals[idx]);
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
            for idx2 in 0..self.population.len() {
                self.population[idx2].individuals[candidates[ridx].idx] =
                    cand_vec[idx2] + rng.gen_range(-self.settings.sigma..self.settings.sigma);
            }
        }

        for idx in 0..self.population.len() {
            // Shuffle
            self.population[idx].individuals.shuffle(&mut rng);
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
