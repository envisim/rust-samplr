use crate::srs::SrsWor;
use envisim_samplr_utils::{
    container::Container,
    kd_tree::{midpoint_slide, Node, Searcher},
    matrix::{Matrix, OperateMatrix, RefMatrix},
    random_generator::RandomGenerator,
};

pub trait CubeMethodVariant<'a, R>
where
    R: RandomGenerator,
{
    fn select_units(
        &mut self,
        candidates: &mut Vec<usize>,
        container: &Container<'a, R>,
        n_units: usize,
    );
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool>;
}

pub struct CubeMethodSampler<'a, R, T>
where
    R: RandomGenerator,
    T: CubeMethodVariant<'a, R>,
{
    container: Box<Container<'a, R>>,
    variant: Box<T>,
    candidates: Vec<usize>,
    adjusted_data: Box<Matrix>,
    candidate_data: Box<Matrix>,
}

pub struct CubeMethod {}
pub struct LocalCubeMethod<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<Searcher>,
}

impl CubeMethod {
    pub fn new<'a, R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
    ) -> CubeMethodSampler<'a, R, Self>
    where
        R: RandomGenerator,
    {
        CubeMethodSampler::new(
            Box::new(Container::new(rand, probabilities, eps)),
            Box::new(CubeMethod {}),
            balancing_data,
        )
    }

    #[inline]
    pub fn sample<'a, R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
    ) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        Self::new(rand, probabilities, eps, balancing_data)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl<'a> LocalCubeMethod<'a> {
    pub fn new<R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
        spreading_data: &'a RefMatrix,
        bucket_size: usize,
    ) -> CubeMethodSampler<'a, R, Self>
    where
        R: RandomGenerator,
    {
        let container = Box::new(Container::new(rand, probabilities, eps));
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            spreading_data,
            container.indices(),
        ));
        let searcher = Box::new(Searcher::new(&tree, balancing_data.ncol()));

        CubeMethodSampler::new(
            container,
            Box::new(LocalCubeMethod {
                tree: tree,
                searcher: searcher,
            }),
            balancing_data,
        )
    }

    #[inline]
    pub fn sample<R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
        spreading_data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        Self::new(
            rand,
            probabilities,
            eps,
            balancing_data,
            spreading_data,
            bucket_size,
        )
        .sample()
        .get_sorted_sample()
        .to_vec()
    }
}

impl<'a, R, T> CubeMethodSampler<'a, R, T>
where
    R: RandomGenerator,
    T: CubeMethodVariant<'a, R>,
{
    fn new(
        container: Box<Container<'a, R>>,
        variant: Box<T>,
        balancing_data: &'a RefMatrix,
    ) -> Self {
        let (b_nrow, b_ncol) = balancing_data.dim();
        assert!(b_nrow == container.population_size());
        let mut adjusted_data = Box::new(Matrix::new(balancing_data.data(), b_nrow));

        for i in 0..b_nrow {
            let p = container.probabilities()[i];
            for j in 0..b_ncol {
                adjusted_data[(i, j)] /= p;
            }
        }

        CubeMethodSampler {
            container: container,
            variant: variant,
            candidates: Vec::<usize>::with_capacity(20),
            adjusted_data: adjusted_data,
            candidate_data: Box::new(Matrix::new_fill(0.0, (b_ncol, b_ncol + 1))),
        }
    }
    #[inline]
    pub fn sample(&mut self) -> &mut Self {
        self.run_flight().run_landing()
    }
    fn run_flight(&mut self) -> &mut Self {
        let b_cols = self.adjusted_data.ncol();
        assert_eq!(b_cols, self.candidate_data.nrow());

        while self.container.indices().len() > b_cols {
            self.variant
                .select_units(&mut self.candidates, &self.container, b_cols + 1);
            self.set_candidate_data().update_probabilities();
        }

        self
    }
    fn run_landing(&mut self) -> &mut Self {
        let b_cols = self.adjusted_data.ncol();
        assert!(
            self.container.indices().len() <= b_cols,
            "landing phase committed early: {} units remaining, with {} cols",
            self.container.indices().len(),
            b_cols,
        );

        while self.container.indices().len() > 1 {
            let number_of_remaining_units = self.container.indices().len();
            unsafe {
                self.candidate_data
                    .resize(number_of_remaining_units - 1, number_of_remaining_units);
            }

            self.candidates.clear();
            self.candidates
                .extend_from_slice(self.container.indices().list());
            self.set_candidate_data().update_probabilities();
        }

        if let Some(id) = self.container.update_last_unit() {
            self.variant.decide_unit(&mut self.container, id);
        }

        self
    }
    #[inline]
    fn set_candidate_data(&mut self) -> &mut Self {
        let b_cols = self.candidates.len() - 1;
        assert_eq!(self.candidate_data.dim(), (b_cols, self.candidates.len()));

        for (i, &id) in self.candidates.iter().enumerate() {
            for j in 0..b_cols {
                self.candidate_data[(j, i)] = self.adjusted_data[(id, j)];
            }
        }

        self
    }
    #[inline]
    fn update_probabilities(&mut self) {
        let uvec = find_vector_in_null_space(&mut self.candidate_data);
        let mut lambdas = (f64::MAX, f64::MAX);

        self.candidates
            .iter()
            .map(|&id| self.container.probabilities()[id])
            .zip(uvec.iter())
            .for_each(|(prob, &uval)| {
                let lvals = ((prob / uval).abs(), ((1.0 - prob) / uval).abs());

                if uval >= 0.0 {
                    lambdas.0 = lambdas.0.min(lvals.1);
                    lambdas.1 = lambdas.1.min(lvals.0);
                } else {
                    lambdas.0 = lambdas.0.min(lvals.0);
                    lambdas.1 = lambdas.1.min(lvals.1);
                }
            });

        let lambda = if self.container.random().one_of_f64(lambdas.0, lambdas.1) {
            lambdas.0
        } else {
            -lambdas.1
        };

        for (i, &id) in self.candidates.iter().enumerate() {
            self.container.probabilities_mut()[id] += lambda * uvec[i];
            self.variant.decide_unit(&mut self.container, id);
        }
    }
    #[inline]
    pub fn get_sample(&mut self) -> &[usize] {
        self.container.sample().get()
    }
    #[inline]
    pub fn get_sorted_sample(&mut self) -> &[usize] {
        self.container.sample_mut().sort().get()
    }
}

impl<'a, R> CubeMethodVariant<'a, R> for CubeMethod
where
    R: RandomGenerator,
{
    fn select_units(
        &mut self,
        candidates: &mut Vec<usize>,
        container: &Container<'a, R>,
        n_units: usize,
    ) {
        assert!(container.indices().len() >= n_units);
        candidates.clear();
        candidates.extend_from_slice(&container.indices().list()[0..n_units]);
    }
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id)
    }
}

impl<'a, R> CubeMethodVariant<'a, R> for LocalCubeMethod<'a>
where
    R: RandomGenerator,
{
    fn select_units(
        &mut self,
        candidates: &mut Vec<usize>,
        container: &Container<'a, R>,
        n_units: usize,
    ) {
        assert!(n_units > 1);
        assert!(container.indices().len() >= n_units);
        candidates.clear();

        if container.indices().len() == n_units {
            candidates.extend_from_slice(&container.indices().list());
            return;
        }

        // Draw the first unit at random
        let id1 = *container.indices_random().unwrap();
        candidates.push(id1);

        // Find the neighbours of this first unit
        self.searcher.find_neighbours_of_id(&self.tree, id1);

        // Add all neighbours, if no equals
        if self.searcher.neighbours().len() == n_units - 1 {
            candidates.extend_from_slice(self.searcher.neighbours());
            return;
        }

        let mut i: usize = 0;
        let maximum_distance = self
            .searcher
            .distance_k(self.searcher.neighbours().len() - 1);

        // Add all neighbours that are not on maximum distance
        while i < n_units - 1 && self.searcher.distance_k(i) < maximum_distance {
            let id = self.searcher.neighbours()[i];
            candidates.push(id);
            i += 1;
        }

        // Randomly add neighbours on the maximum distance
        for k in SrsWor::sample(
            container.random(),
            n_units - candidates.len(),
            self.searcher.neighbours().len() - i,
        )
        .iter()
        {
            candidates.push(self.searcher.neighbours()[i + k]);
        }
    }
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).and_then(|r| {
            self.tree.remove_unit(id);
            Some(r)
        })
    }
}

/// Finds a vector in null space of a (n-1)*n matrix. The matrix is
/// mutated into rref.
fn find_vector_in_null_space(mat: &mut Matrix) -> Vec<f64> {
    let (nrow, ncol) = mat.dim();
    assert!(nrow > 0);
    assert!(nrow == ncol - 1);

    mat.reduced_row_echelon_form();
    // If (0, 0) == 0.0, then the we have big problems
    assert!(mat[(0, 0)] != 0.0);

    let mut v = vec![1.0; ncol];

    // If (n-1, n-1) = 1.0, then we have linearly independent rows,
    // and the form of the matrix is an identity matrix with the parts
    // of the null space vector in the last column
    if mat[(nrow - 1, nrow - 1)] == 1.0 {
        for i in 0..nrow {
            v[i] = -mat[(i, ncol - 1)];
        }

        return v;
    }

    // If we have some linearly dependent rows, we must take a slower
    // route
    for k in 1..ncol {
        v[k] = if k % 2 == 0 { -1.0 } else { 1.0 };
    }

    for i in 0..nrow {
        let mut lead: usize = 0;

        while lead < ncol && mat[(i, lead)] != 1.0 {
            lead += 1;
        }

        if lead == ncol {
            continue;
        }

        v[lead] = 0.0;

        for k in (lead + 1)..ncol {
            v[lead] -= v[k] * mat[(i, k)];
        }
    }

    return v;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_delta, data_10_2, EPS, RAND00};

    #[test]
    fn cube() {
        let balmat = RefMatrix::new(
            &[
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, //
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            ],
            10,
        );
        let (_sprmat, prob) = data_10_2();

        let mut cube = CubeMethod::new(&RAND00, &prob, EPS, &balmat);
        cube.run_flight();
        assert!(cube.container.indices().len() < 3);

        let s = CubeMethod::sample(&RAND00, &prob, EPS, &balmat);
        assert!(s.len() == 2);
    }

    #[test]
    fn local_cube() {
        let balmat = RefMatrix::new(
            &[
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, //
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            ],
            10,
        );
        let (sprmat, prob) = data_10_2();

        let mut cube = LocalCubeMethod::new(&RAND00, &prob, EPS, &balmat, &sprmat, 2);
        cube.run_flight();
        assert!(cube.container.indices().len() < 3);

        let s = LocalCubeMethod::sample(&RAND00, &prob, EPS, &balmat, &sprmat, 2);
        assert!(s.len() == 2);
    }

    #[test]
    fn null() {
        let mut mat1 = Matrix::new(
            &[
                1.0, 2.0, 3.0, 1.0, 5.0, 10.0, 1.0, 5.0, 10.0, 1.0, 5.0, 10.0,
            ],
            3,
        );
        mat1.reduced_row_echelon_form();
        assert!(mat1.data() == [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        let mat1_nullvec = find_vector_in_null_space(&mut mat1);
        let res1 = mat1.prod_vec(&mat1_nullvec);
        assert_delta!(res1[0], 0.0, EPS);
        assert_delta!(res1[1], 0.0, EPS);
        assert_delta!(res1[2], 0.0, EPS);

        let mut mat2 = Matrix::new(
            &[
                1.0, 2.0, 3.0, 1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 1.0, 5.0, 11.0,
            ],
            3,
        );
        mat2.reduced_row_echelon_form();
        assert!(&mat2.data()[0..9] == vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_delta!(mat2[(0, 3)], -2.5, EPS);
        assert_delta!(mat2[(1, 3)], 1.833333333333333, EPS);
        assert_delta!(mat2[(2, 3)], 0.166666666666667, EPS);
        let mat2_nullvec = find_vector_in_null_space(&mut mat2);
        let res2 = mat2.prod_vec(&mat2_nullvec);
        assert_delta!(res2[0], 0.0, EPS);
        assert_delta!(res2[1], 0.0, EPS);
        assert_delta!(res2[2], 0.0, EPS);
    }
}
