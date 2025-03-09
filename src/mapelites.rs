/*
 * Implements a type of MAP-Elites as described in:
 * "Illuminating search spaces by mapping elites" 2015
 * by Jean-Baptiste Mouret and Jeff Clune
 * https://arxiv.org/pdf/1504.04909.pdf
 *
 */

use crate::vectorizable::Vectorizable;
use rand::prelude::*;
use rand::rng;
use skiplist::skipmap::SkipMap;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct MAPElitesSettings {
    batch_size: usize,
    sample_strategy: MAPElitesSampleStrategy,
    sigma: f64,
    max_stored_elites: usize,
    delete_n_when_full: usize,
}

impl MAPElitesSettings {
    pub fn default() -> Self {
        MAPElitesSettings {
            batch_size: 32,
            sigma: 0.1,
            sample_strategy: MAPElitesSampleStrategy::Random,
            max_stored_elites: 500000,
            delete_n_when_full: 10000,
        }
    }

    pub fn sigma(mut self, sigma: f64) -> Self {
        if sigma <= 0.0 {
            panic!("sigma must be greater than 0");
        }
        self.sigma = sigma;
        self
    }

    pub fn sample_strategy(mut self, sample_strategy: MAPElitesSampleStrategy) -> Self {
        self.sample_strategy = sample_strategy;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        if batch_size == 0 {
            panic!("batch_size must be greater than 0");
        }
        self.batch_size = batch_size;
        self
    }
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MAPElitesSampleStrategy {
    Random,
    BiasTopOnes,
}

#[derive(Debug)]
pub struct MAPElites<T, EliteKey> {
    settings: MAPElitesSettings,

    // If non-empty, then these are returned next from ask()
    next_testables: VecDeque<T>,
    elites: SkipMap<EliteKey, (T, f64)>,
}

impl<T: Clone, EliteKey: Clone + Ord> Clone for MAPElites<T, EliteKey> {
    fn clone(&self) -> Self {
        let mut elites = SkipMap::new();
        for (key, value) in self.elites.iter() {
            elites.insert(key.clone(), value.clone());
            if elites.len() >= self.settings.max_stored_elites {
                let mut rng = rng();
                for _ in 0..self.settings.delete_n_when_full {
                    elites.remove_index(rng.random_range(0..elites.len()));
                }
            }
        }
        Self {
            settings: self.settings.clone(),
            next_testables: self.next_testables.clone(),
            elites,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MAPElitesCandidate<T, EliteKey> {
    item: T,
    score: f64,
    elite_key: Option<EliteKey>,
}

impl<T: Clone, EliteKey> MAPElitesCandidate<T, EliteKey> {
    fn new(item: T) -> Self {
        Self {
            item,
            score: 0.0,
            elite_key: None,
        }
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn set_score(&mut self, score: f64) {
        self.score = score;
    }

    pub fn set_elite_key(&mut self, elite_key: EliteKey) {
        self.elite_key = Some(elite_key);
    }

    pub fn item(&self) -> &T {
        &self.item
    }
}

pub struct IterElites<'a, EliteKey, T> {
    iter: skiplist::skipmap::Iter<'a, EliteKey, (T, f64)>,
}

impl<'a, EliteKey: Clone, T> Iterator for IterElites<'a, EliteKey, T> {
    type Item = (EliteKey, f64);

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            None => None,
            Some((key, value)) => Some((key.clone(), value.1)),
        }
    }
}

impl<T: Clone + Vectorizable, EliteKey: Clone + Ord> MAPElites<T, EliteKey> {
    pub fn new(initials: &[T], settings: &MAPElitesSettings) -> Self {
        if initials.is_empty() {
            panic!("MAPElites must be initialized with at least one individual");
        }

        let mut next_testables = VecDeque::new();
        for i in initials.iter() {
            next_testables.push_back(i.clone());
        }

        Self {
            settings: settings.clone(),
            next_testables,
            elites: SkipMap::new(),
        }
    }

    /// Iterate over all elites. Gives the elites and their score.
    pub fn iter_elites<'a>(&'a self) -> IterElites<'a, EliteKey, T> {
        IterElites {
            iter: self.elites.iter(),
        }
    }

    fn generate_testable(&self) -> T {
        if self.elites.is_empty() {
            panic!("MAPElites must be initialized with at least one individual");
        }
        // Pick random elite
        let mut rng = rng();
        let elite: &(T, f64) = match self.settings.sample_strategy {
            MAPElitesSampleStrategy::Random => &self.elites[rng.random_range(0..self.elites.len())],
            MAPElitesSampleStrategy::BiasTopOnes => {
                let mut best_ones: Vec<(usize, f64)> = Vec::with_capacity(self.elites.len());
                for idx in 0..self.elites.len() {
                    best_ones.push((idx, self.elites[idx].1));
                }
                best_ones.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let idx = rng.random_range(0..self.elites.len());
                let idx2 = rng.random_range(0..=idx);
                let idx3 = rng.random_range(0..=idx2);
                let idx4 = rng.random_range(0..=idx3);
                let idx5 = rng.random_range(0..=idx4);
                &self.elites[best_ones[idx5].0]
            }
        };
        let (mut vec, ctx) = elite.0.to_vec();
        for v in vec.iter_mut() {
            *v += rng.random_range(-self.settings.sigma..self.settings.sigma);
        }
        let item = T::from_vec(&vec, &ctx);
        item
    }

    pub fn ask(&mut self) -> Vec<MAPElitesCandidate<T, EliteKey>> {
        if !self.next_testables.is_empty() {
            let mut result = Vec::with_capacity(self.next_testables.len());
            for t in self.next_testables.drain(..) {
                result.push(MAPElitesCandidate::new(t));
            }
            return result;
        }
        let mut result = Vec::with_capacity(self.settings.batch_size);
        for _ in 0..self.settings.batch_size {
            let candidate = self.generate_testable();
            result.push(MAPElitesCandidate::new(candidate));
        }
        result
    }

    pub fn tell(&mut self, candidate: MAPElitesCandidate<T, EliteKey>) {
        if candidate.elite_key.is_none() {
            panic!("MAPElitesCandidate must have an elite key");
        }
        let elite_key = candidate.elite_key.unwrap();
        match self.elites.get_mut(&elite_key) {
            None => {
                self.elites
                    .insert(elite_key, (candidate.item, candidate.score));
            }
            Some(existing) => {
                if existing.1 > candidate.score {
                    *existing = (candidate.item, candidate.score);
                }
            }
        }
    }
}
