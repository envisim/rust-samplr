use crate::uniform_random::{DrawRandom, RandomGenerator};

#[derive(Clone)]
pub struct Indices {
    list: Vec<usize>,
    reverse_list: Vec<usize>,
    capacity: usize,
}

impl Indices {
    #[inline]
    pub fn new(len: usize) -> Self {
        Indices {
            list: Vec::<usize>::with_capacity(len),
            reverse_list: Vec::<usize>::with_capacity(len),
            capacity: len,
        }
    }

    #[inline]
    pub fn new_fill(len: usize) -> Self {
        let mut list = Indices::new(len);
        list.fill(len);
        list
    }

    #[inline]
    pub fn fill(&mut self, to: usize) -> &mut Self {
        self.list.clear();
        self.reverse_list.clear();

        for k in 0usize..to {
            self.list.push(k);
            self.reverse_list.push(k);
        }

        self
    }

    #[inline]
    pub fn get_at(&self, k: usize) -> usize {
        self.list[k]
    }

    #[inline]
    pub fn get_first(&self) -> usize {
        self.list[0]
    }

    #[inline]
    pub fn get_at_id(&self, id: usize) -> usize {
        self.reverse_list[id]
    }

    #[inline]
    pub fn get_last(&self) -> usize {
        *self.list.last().unwrap()
    }

    #[inline]
    pub fn get_random(&self, rand: &RandomGenerator) -> usize {
        self.list.draw(rand)
    }

    #[inline]
    pub fn includes(&self, id: usize) -> bool {
        self.reverse_list[id] < self.list.len()
    }

    #[inline]
    pub fn insert(&mut self, id: usize) -> &mut Self {
        assert!(id < self.capacity);

        let len: usize = self.list.len();

        if self.reverse_list[id] < len {
            // id already is in list
            return self;
        }

        self.reverse_list[id] = len;
        self.list.push(id);
        self
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.list.len()
    }

    #[inline]
    pub fn remove(&mut self, id: usize) -> &mut Self {
        assert!(id < self.capacity);

        let len: usize = self.list.len();
        let k: usize = self.reverse_list[id];

        assert!(k < len);

        if k == len - 1 {
            self.list.pop();
            self.reverse_list[id] = usize::MAX;
            return self;
        }

        self.list.swap_remove(k);
        self.reverse_list[id] = usize::MAX;
        self.reverse_list[self.list[k]] = k;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill() {
        let mut il = Indices::new(4);
        il.fill(4);
        let list: Vec<usize> = vec![0, 1, 2, 3];
        assert_eq!(list, il.list);
    }
}
