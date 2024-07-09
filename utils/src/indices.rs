use crate::generate_random::GenerateRandom;

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
    pub fn fill(&mut self, to: usize) {
        self.list.clear();
        self.reverse_list.clear();

        for k in 0usize..to {
            self.list.push(k);
            self.reverse_list.push(k);
        }
    }

    #[inline]
    pub fn get(&self, k: usize) -> Option<&usize> {
        self.list.get(k)
    }

    #[inline]
    pub fn get_first(&self) -> Option<&usize> {
        self.list.first()
    }

    #[inline]
    pub fn get_last(&self) -> Option<&usize> {
        self.list.last()
    }

    pub fn get_list(&self) -> &[usize] {
        &self.list
    }

    #[inline]
    pub fn get_random<R: GenerateRandom>(&self, rand: &R) -> Option<&usize> {
        rand.random_get(&self.list)
    }

    #[inline]
    pub fn includes(&self, id: usize) -> bool {
        self.reverse_list[id] < self.list.len()
    }

    #[inline]
    pub fn insert(&mut self, id: usize) {
        assert!(id < self.capacity);

        let len: usize = self.list.len();

        if self.reverse_list[id] < len {
            // id already is in list
            return;
        }

        self.reverse_list[id] = len;
        self.list.push(id);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.list.len()
    }

    #[inline]
    pub fn remove(&mut self, id: usize) {
        assert!(id < self.capacity);

        let len: usize = self.list.len();
        let k: usize = self.reverse_list[id];

        assert!(k < len);

        if k == len - 1 {
            self.list.pop();
            self.reverse_list[id] = usize::MAX;
            return;
        }

        self.list.swap_remove(k);
        self.reverse_list[id] = usize::MAX;
        self.reverse_list[self.list[k]] = k;
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
