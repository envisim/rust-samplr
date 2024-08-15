use crate::utils::random_element;
use rand::Rng;
use rustc_hash::{FxBuildHasher, FxHashMap};
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum IndicesError {
    #[error("cannot access non-existing index {0}")]
    GhostIndex(usize),
    #[error("cannot access out-of-bounds k-index {0}")]
    OutOfBoundsK(usize),
}

pub struct Indices {
    list: Vec<usize>,
    indices: FxHashMap<usize, usize>,
}

impl Indices {
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Indices {
            list: Vec::<usize>::with_capacity(capacity),
            indices: FxHashMap::<usize, usize>::with_capacity_and_hasher(capacity, FxBuildHasher),
        }
    }

    #[inline]
    pub fn with_fill(length: usize) -> Self {
        Indices {
            list: (0..length).collect::<Vec<usize>>(),
            indices: (0..length)
                .map(|v| (v, v))
                .collect::<FxHashMap<usize, usize>>(),
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.list.clear();
        self.indices.clear();
    }

    #[inline]
    pub fn list(&self) -> &[usize] {
        &self.list
    }

    #[inline]
    pub fn get(&self, k: usize) -> Option<&usize> {
        self.list.get(k)
    }

    #[inline]
    pub fn first(&self) -> Option<&usize> {
        self.list.first()
    }

    #[inline]
    pub fn last(&self) -> Option<&usize> {
        self.list.last()
    }

    #[inline]
    pub fn draw<R>(&self, rng: &mut R) -> Option<&usize>
    where
        R: Rng + ?Sized,
    {
        random_element(rng, &self.list)
    }

    #[inline]
    pub fn contains(&self, id: usize) -> bool {
        self.indices.contains_key(&id)
    }

    #[inline]
    pub fn insert(&mut self, id: usize) -> Result<usize, IndicesError> {
        if self.contains(id) {
            return Err(IndicesError::GhostIndex(id));
        }

        self.list.push(id);
        let k = self.list.len() - 1;
        self.indices.insert(id, k);
        Ok(k)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.list.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    #[inline]
    pub fn remove(&mut self, id: usize) -> Result<(), IndicesError> {
        match self.indices.remove(&id) {
            Some(k) => {
                self.list.swap_remove(k);

                if k != self.list.len() {
                    *self.indices.get_mut(&self.list[k]).unwrap() = k;
                }

                Ok(())
            }
            None => Err(IndicesError::GhostIndex(id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let il = Indices::new(10);
        assert!(il.list.capacity() >= 10);
        assert!(il.indices.capacity() >= 10);
    }

    #[test]
    fn with_fill() {
        let il = Indices::with_fill(4);
        assert_eq!(il.list(), vec![0, 1, 2, 3]);
    }
}
