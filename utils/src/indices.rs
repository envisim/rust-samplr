use crate::random_generator::RandomGenerator;
use std::collections::HashMap;

pub struct Indices {
    list: Vec<usize>,
    indices: HashMap<usize, usize>,
}

impl Indices {
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Indices {
            list: Vec::<usize>::with_capacity(capacity),
            indices: HashMap::<usize, usize>::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn with_fill(length: usize) -> Self {
        Indices {
            list: Vec::<usize>::from_iter(0..length),
            indices: HashMap::<usize, usize>::from_iter((0..length).map(|v| (v, v))),
        }
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
    pub fn random<R>(&self, rand: &R) -> Option<&usize>
    where
        R: RandomGenerator,
    {
        rand.rslice(&self.list)
    }

    #[inline]
    pub fn contains(&self, id: usize) -> bool {
        self.indices.contains_key(&id)
    }

    #[inline]
    pub fn insert(&mut self, id: usize) -> bool {
        if self.contains(id) {
            return false;
        }

        self.list.push(id);
        self.indices.insert(id, self.list.len() - 1);
        true
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.list.len()
    }

    #[inline]
    pub fn remove(&mut self, id: usize) -> bool {
        match self.indices.remove(&id) {
            Some(k) => {
                self.list.swap_remove(k);

                if k != self.list.len() {
                    *self.indices.get_mut(&self.list[k]).unwrap() = k;
                }

                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill() {
        let mut il = Indices::with_fill(4);
        let list: Vec<usize> = vec![0, 1, 2, 3];
        assert_eq!(list, il.list);

        assert!(il.remove(1));
        assert!(!il.remove(1));
        assert_eq!(il.last().unwrap(), &2);
    }
}
