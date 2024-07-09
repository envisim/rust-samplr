use envisim_samplr_utils::{
    generate_random::GenerateRandom,
    kd_tree::{midpoint_slide, SearcherForNeighbours, Tree, TreeSearch},
    matrix::{Matrix, OperateMatrix, RefMatrix},
    sampling_controller::*,
};

trait RunCubeMethod<R: GenerateRandom, C: AccessBaseController<R>> {
    fn controller(&self) -> &C;
    fn controller_mut(&mut self) -> &mut C;
    #[inline]
    fn draw_units(&mut self) {
        let ncols = self.adjusted_data().ncol();
        let remaining_units = self.controller().indices().len();
        let maximum_size = ncols + 1;
        self.candidates_mut().clear();

        if remaining_units < maximum_size {
            panic!("units should not be drawn in landing phase");
        }

        for i in 0..maximum_size {
            let id = *self.controller().indices().get(i).unwrap();
            self.candidates_mut().push(id);
        }
    }
    fn adjusted_data(&self) -> &Matrix;
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.controller_mut().decide_unit(id)
    }
    fn candidate_data_mut(&mut self) -> &mut Matrix;
    fn candidates(&self) -> &[usize];
    fn candidates_mut(&mut self) -> &mut Vec<usize>;
    #[inline]
    fn update_probabilities(&mut self) {
        let uvec = find_vector_in_null_space(self.candidate_data_mut());

        let mut lambda1 = f64::MAX;
        let mut lambda2 = f64::MAX;

        for i in 0..self.candidates().len() {
            let lval1 = (self.controller().probability(self.candidates()[i]) / uvec[i]).abs();
            let lval2 =
                ((1.0 - self.controller().probability(self.candidates()[i])) / uvec[i]).abs();

            if uvec[i] >= 0.0 {
                if lambda1 > lval2 {
                    lambda1 = lval2;
                }
                if lambda2 > lval1 {
                    lambda2 = lval1;
                }
            } else {
                if lambda1 > lval1 {
                    lambda1 = lval1;
                }
                if lambda2 > lval2 {
                    lambda2 = lval2;
                }
            }
        }

        let lambda = if self
            .controller()
            .random()
            .random_float_scale(lambda1 + lambda2)
            < lambda2
        {
            lambda1
        } else {
            -lambda2
        };

        for i in 0..self.candidates().len() {
            let id = self.candidates()[i];
            *self.controller_mut().probability_mut(id) += lambda * uvec[i];
            self.decide_unit(id);
        }
    }
    #[inline]
    fn run_flight(&mut self) {
        let b_cols = self.adjusted_data().ncol();
        if self.controller().indices().len() < b_cols + 1 {
            return;
        }

        while self.controller().indices().len() > b_cols {
            self.draw_units();
            assert_eq!(self.candidates().len(), b_cols + 1);

            for i in 0..self.candidates().len() {
                for j in 0..b_cols {
                    // candidate_data should have a unit per column
                    let id = self.candidates()[i];
                    self.candidate_data_mut()[(j, i)] = self.adjusted_data()[(id, j)];
                }
            }

            self.update_probabilities();
        }
    }
    #[inline]
    fn run_landing(&mut self) {
        let b_cols = self.adjusted_data().ncol();
        assert!(
            self.controller().indices().len() <= b_cols,
            "landing phase committed early: {} units remaining, with {} cols",
            self.controller().indices().len(),
            b_cols,
        );

        while self.controller().indices().len() > 1 {
            let number_of_remaining_units = self.controller().indices().len();
            self.candidates_mut().clear();
            unsafe {
                self.candidate_data_mut()
                    .resize(number_of_remaining_units - 1, number_of_remaining_units)
            };

            for i in 0..number_of_remaining_units {
                let id = *self
                    .controller()
                    .indices()
                    .get(i)
                    .expect("there should be remaining units");
                self.candidates_mut().push(id);
                for j in 0..(number_of_remaining_units - 1) {
                    self.candidate_data_mut()[(j, i)] = self.adjusted_data()[(id, j)];
                }
            }

            self.update_probabilities();
        }

        if let Some(id) = self.controller_mut().update_last_unit() {
            self.decide_unit(id);
        }
    }
    #[inline]
    fn run(&mut self) {
        self.run_flight();
        self.run_landing();
    }
}

pub struct CubeMethod<'a, R: GenerateRandom> {
    controller: BaseController<'a, R>,
    adjusted_data: Box<Matrix>,
    candidate_data: Box<Matrix>,
    candidates: Vec<usize>,
}

impl<'a, R: GenerateRandom> CubeMethod<'a, R> {
    pub fn new(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
    ) -> Self {
        let (b_nrow, b_ncol) = balancing_data.dim();
        assert!(b_nrow == probabilities.len());
        let mut adjusted_data = Box::new(Matrix::new(balancing_data.data(), b_nrow));

        for i in 0..b_nrow {
            let p = probabilities[i];
            for j in 0..b_ncol {
                adjusted_data[(i, j)] /= p;
            }
        }

        Self {
            controller: BaseController::new(rand, probabilities, eps),
            adjusted_data: adjusted_data,
            candidate_data: Box::new(Matrix::new_fill(0.0, (b_ncol, b_ncol + 1))),
            candidates: Vec::<usize>::with_capacity(b_ncol + 1),
        }
    }
    pub fn draw(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
    ) -> Vec<usize> {
        let mut cube = Self::new(rand, probabilities, eps, balancing_data);
        cube.run();
        cube.controller_mut().sample_sort();
        cube.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunCubeMethod<R, BaseController<'a, R>> for CubeMethod<'a, R> {
    #[inline]
    fn controller(&self) -> &BaseController<'a, R> {
        &self.controller
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut BaseController<'a, R> {
        &mut self.controller
    }
    #[inline]
    fn adjusted_data(&self) -> &Matrix {
        &self.adjusted_data
    }
    #[inline]
    fn candidate_data_mut(&mut self) -> &mut Matrix {
        &mut self.candidate_data
    }
    #[inline]
    fn candidates(&self) -> &[usize] {
        &self.candidates
    }
    #[inline]
    fn candidates_mut(&mut self) -> &mut Vec<usize> {
        &mut self.candidates
    }
}

pub struct LocalCubeMethod<'a, R: GenerateRandom> {
    tree: Tree<'a, R, SearcherForNeighbours>,
    adjusted_data: Box<Matrix>,
    candidate_data: Box<Matrix>,
    candidates: Vec<usize>,
}

impl<'a, R: GenerateRandom> LocalCubeMethod<'a, R> {
    pub fn new(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
        spreading_data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Self {
        let (b_nrow, b_ncol) = balancing_data.dim();
        assert!(b_nrow == probabilities.len());
        let mut adjusted_data = Box::new(Matrix::new(balancing_data.data(), b_nrow));

        for i in 0..b_nrow {
            let p = probabilities[i];
            for j in 0..b_ncol {
                adjusted_data[(i, j)] /= p;
            }
        }

        Self {
            tree: Tree::new(
                DataController::new(rand, probabilities, eps, spreading_data),
                SearcherForNeighbours::new(spreading_data.dim(), balancing_data.ncol()),
                midpoint_slide,
                bucket_size,
            ),
            adjusted_data: adjusted_data,
            candidate_data: Box::new(Matrix::new_fill(0.0, (b_ncol, b_ncol + 1))),
            candidates: Vec::<usize>::with_capacity(b_ncol + 1),
        }
    }
    pub fn draw(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
        spreading_data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize> {
        let mut cube = Self::new(
            rand,
            probabilities,
            eps,
            balancing_data,
            spreading_data,
            bucket_size,
        );
        cube.run();
        cube.controller_mut().sample_sort();
        cube.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunCubeMethod<R, DataController<'a, R>> for LocalCubeMethod<'a, R> {
    #[inline]
    fn controller(&self) -> &DataController<'a, R> {
        self.tree.controller()
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut DataController<'a, R> {
        self.tree.controller_mut()
    }
    #[inline]
    fn adjusted_data(&self) -> &Matrix {
        &self.adjusted_data
    }
    #[inline]
    fn candidate_data_mut(&mut self) -> &mut Matrix {
        &mut self.candidate_data
    }
    #[inline]
    fn candidates(&self) -> &[usize] {
        &self.candidates
    }
    #[inline]
    fn candidates_mut(&mut self) -> &mut Vec<usize> {
        &mut self.candidates
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.tree.decide_unit(id)
    }
    fn draw_units(&mut self) {
        let ncols = self.adjusted_data().ncol();
        let remaining_units = self.controller().indices().len();
        self.candidates.clear();

        if remaining_units < ncols + 1 {
            panic!("units should not be drawn in landing phase");
        } else if remaining_units == ncols + 1 {
            for i in 0..remaining_units {
                let id = *self.controller().indices().get(i).unwrap();
                self.candidates.push(id);
            }

            return;
        }

        // Draw the first unit at random
        let id1 = *self
            .controller()
            .get_random_index()
            .expect("indices should not be empty");
        self.candidates.push(id1);

        // Find the neighbours of this first unit
        self.tree.find_neighbours_of_id(id1);

        // Add all neighbours, if no equals
        if self.tree.searcher().neighbours().len() == ncols {
            for i in 0..ncols {
                let id = self.tree.searcher().neighbour_k(i);
                self.candidates.push(id);
            }

            return;
        }

        let mut i: usize = 0;
        let maximum_distance = self.tree.searcher().max_distance();

        // Add all neighbours that are not on maximum distance
        while i < ncols && self.tree.searcher().distance_k(i) < maximum_distance {
            let id = self.tree.searcher().neighbour_k(i);
            self.candidates.push(id);
            i += 1;
        }

        // Randomly add neighbours on the maximum distance
        while i < ncols {
            let k = self.controller().random().random_usize_scale(ncols - i);
            let id = self.tree.searcher().neighbour_k(k + i);
            self.candidates.push(id);
            unsafe { self.tree.searcher_mut().neighbours_mut().swap(i, k + i) };
            i += 1;
        }
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
    use envisim_samplr_utils::generate_random::StaticRandom;

    macro_rules! assert_delta {
        ($a:expr,$b:expr,$d:expr) => {
            assert!(
                ($a - $b).abs() < $d,
                "|{} - {}| = {} > {}",
                $a,
                $b,
                ($a - $b).abs(),
                $d
            );
        };
    }

    const RAND00: StaticRandom = StaticRandom::new(0.0);

    #[test]
    fn cube() {
        let balmat = RefMatrix::new(
            &[
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, //
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            ],
            10,
        );
        let prob = vec![0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2];

        let mut cube = CubeMethod::new(&RAND00, &prob, 1e-12, &balmat);
        cube.run_flight();
        assert!(cube.controller().indices().len() < 3);

        let s = CubeMethod::draw(&RAND00, &prob, 1e-12, &balmat);
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
        let sprmat = RefMatrix::new(
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
                9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, //
            ],
            10,
        );

        let prob = vec![0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2];

        let mut cube = LocalCubeMethod::new(&RAND00, &prob, 1e-12, &balmat, &sprmat, 2);
        cube.run_flight();
        assert!(cube.controller().indices().len() < 3);

        let s = LocalCubeMethod::draw(&RAND00, &prob, 1e-12, &balmat, &sprmat, 2);
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
        assert_delta!(res1[0], 0.0, 1e-12);
        assert_delta!(res1[1], 0.0, 1e-12);
        assert_delta!(res1[2], 0.0, 1e-12);

        let mut mat2 = Matrix::new(
            &[
                1.0, 2.0, 3.0, 1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 1.0, 5.0, 11.0,
            ],
            3,
        );
        mat2.reduced_row_echelon_form();
        assert!(&mat2.data()[0..9] == vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_delta!(mat2[(0, 3)], -2.5, 1e-12);
        assert_delta!(mat2[(1, 3)], 1.833333333333333, 1e-12);
        assert_delta!(mat2[(2, 3)], 0.166666666666667, 1e-12);
        let mat2_nullvec = find_vector_in_null_space(&mut mat2);
        let res2 = mat2.prod_vec(&mat2_nullvec);
        assert_delta!(res2[0], 0.0, 1e-12);
        assert_delta!(res2[1], 0.0, 1e-12);
        assert_delta!(res2[2], 0.0, 1e-12);
    }
}
