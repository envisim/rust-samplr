use crate::srs::SrsWor;
use envisim_samplr_utils::{
    container::Container,
    kd_tree::{midpoint_slide, Node, Searcher},
    matrix::{Matrix, OperateMatrix, RefMatrix},
    random_generator::RandomGenerator,
};
use rustc_hash::FxSeededState;
use std::collections::HashMap;

pub trait CubeMethodVariant<'a, R>
where
    R: RandomGenerator,
{
    fn select_units(
        &mut self,
        candidates: &mut Vec<usize>,
        container: &mut Container<'a, R>,
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
        rand: &'a mut R,
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
        rand: &'a mut R,
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

    pub fn sample_stratified<'a, R>(
        rand: &'a mut R,
        probabilities: &'a [f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
        strata: &'a [i64],
    ) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        let seed = rand.rusize(probabilities.len());
        let container = Box::new(Container::new(rand, probabilities, eps));

        let mut cs = CubeStratified {
            cube: CubeMethodSampler {
                container: container,
                variant: Box::new(CubeMethod {}),
                candidates: Vec::<usize>::with_capacity(20),
                adjusted_data: Box::new(Matrix::new_fill(
                    0.0,
                    (balancing_data.nrow(), balancing_data.ncol() + 1),
                )),
                candidate_data: Box::new(Matrix::new_fill(
                    0.0,
                    (balancing_data.ncol() + 1, balancing_data.ncol() + 2),
                )),
            },
            strata: HashMap::<i64, Vec<usize>, FxSeededState>::with_capacity_and_hasher(
                probabilities.len() / 10,
                FxSeededState::with_seed(seed),
            ),
            probabilities: probabilities,
            balancing_data: balancing_data,
            strata_vec: strata,
            data: None,
        };

        cs.prepare().sample()
    }
}

impl<'a> LocalCubeMethod<'a> {
    pub fn new<R>(
        rand: &'a mut R,
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
        rand: &'a mut R,
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

    pub fn sample_stratified<R>(
        rand: &'a mut R,
        probabilities: &'a [f64],
        eps: f64,
        balancing_data: &'a RefMatrix,
        spreading_data: &'a RefMatrix,
        bucket_size: usize,
        strata: &'a [i64],
    ) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        let seed = rand.rusize(probabilities.len());
        let container = Box::new(Container::new(rand, probabilities, eps));
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            spreading_data,
            container.indices(),
        ));
        let searcher = Box::new(Searcher::new(&tree, balancing_data.ncol() + 1));

        let mut cs = CubeStratified {
            cube: CubeMethodSampler {
                container: container,
                variant: Box::new(LocalCubeMethod {
                    tree: tree,
                    searcher: searcher,
                }),
                candidates: Vec::<usize>::with_capacity(20),
                adjusted_data: Box::new(Matrix::new_fill(
                    0.0,
                    (balancing_data.nrow(), balancing_data.ncol() + 1),
                )),
                candidate_data: Box::new(Matrix::new_fill(
                    0.0,
                    (balancing_data.ncol() + 1, balancing_data.ncol() + 2),
                )),
            },
            strata: HashMap::<i64, Vec<usize>, FxSeededState>::with_capacity_and_hasher(
                probabilities.len() / 10,
                FxSeededState::with_seed(seed),
            ),
            probabilities: probabilities,
            balancing_data: balancing_data,
            strata_vec: strata,
            data: Some((spreading_data, bucket_size)),
        };

        cs.prepare().sample()
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
                .select_units(&mut self.candidates, &mut self.container, b_cols + 1);
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
        container: &mut Container<'a, R>,
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
        container: &mut Container<'a, R>,
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

pub trait CubeStratifier<'a, R>: CubeMethodVariant<'a, R>
where
    R: RandomGenerator,
{
    fn reset_to(
        &mut self,
        container: &mut Container<R>,
        ids: &mut [usize],
        data: Option<(&'a RefMatrix, usize)>,
        n_neighbours: usize,
    );
}

pub struct CubeStratified<'a, R, T>
where
    R: RandomGenerator,
    T: CubeMethodVariant<'a, R>,
{
    cube: CubeMethodSampler<'a, R, T>,
    strata: HashMap<i64, Vec<usize>, FxSeededState>,
    probabilities: &'a [f64],
    balancing_data: &'a RefMatrix<'a>,
    strata_vec: &'a [i64],
    data: Option<(&'a RefMatrix<'a>, usize)>,
}

impl<'a, R> CubeStratifier<'a, R> for CubeMethod
where
    R: RandomGenerator,
{
    fn reset_to(
        &mut self,
        container: &mut Container<R>,
        ids: &mut [usize],
        _data: Option<(&'a RefMatrix, usize)>,
        _n_neighbours: usize,
    ) {
        container.indices_mut().clear();
        for id in ids.iter() {
            container.indices_mut().insert(*id);
        }
    }
}

impl<'a, R> CubeStratifier<'a, R> for LocalCubeMethod<'a>
where
    R: RandomGenerator,
{
    fn reset_to(
        &mut self,
        container: &mut Container<R>,
        ids: &mut [usize],
        data: Option<(&'a RefMatrix, usize)>,
        n_neighbours: usize,
    ) {
        assert!(data.is_some());
        self.searcher.set_n_neighbours(n_neighbours);

        container.indices_mut().clear();
        self.tree = Box::new(Node::new(
            midpoint_slide,
            data.unwrap().1,
            data.unwrap().0,
            ids,
        ));

        for id in ids.iter() {
            container.indices_mut().insert(*id);
        }
    }
}

impl<'a, R, T> CubeStratified<'a, R, T>
where
    R: RandomGenerator,
    T: CubeStratifier<'a, R>,
{
    fn prepare(&mut self) -> &mut Self {
        assert_eq!(self.strata_vec.len(), self.cube.container.population_size());
        assert_eq!(
            self.balancing_data.nrow(),
            self.cube.container.population_size()
        );

        for i in 0..self.cube.container.probabilities().len() {
            if !self.cube.container.indices().contains(i) {
                continue;
            }

            let stratum = self.strata_vec[i];
            match self.strata.get_mut(&stratum) {
                Some(uvec) => {
                    uvec.push(i);
                }
                None => {
                    self.strata.insert(stratum, vec![i]);
                }
            };

            // Order doesn't matter during flight
            self.cube.adjusted_data[(i, self.balancing_data.ncol())] = 1.0;
            for j in 0..self.balancing_data.ncol() {
                self.cube.adjusted_data[(i, j)] =
                    self.balancing_data[(i, j)] / self.probabilities[i];
            }
        }

        self
    }
    fn sample(&mut self) -> Vec<usize> {
        self.flight_per_stratum();
        if self.strata.len() == 0 {
            return self.cube.get_sorted_sample().to_vec();
        }
        self.flight_on_full();
        if self.cube.container.indices().len() == 0 {
            return self.cube.get_sorted_sample().to_vec();
        }
        self.landing_per_stratum();
        self.cube.get_sorted_sample().to_vec()
    }
    fn flight_per_stratum(&mut self) {
        let mut removable_stratums = Vec::<i64>::new();
        for (stratum_key, stratum) in self.strata.iter_mut() {
            self.cube.variant.reset_to(
                &mut self.cube.container,
                stratum,
                self.data,
                self.cube.adjusted_data.ncol() + 1,
            );

            self.cube.run_flight();

            if self.cube.container.indices().len() == 0 {
                removable_stratums.push(*stratum_key);
                continue;
            }

            stratum.clear();
            stratum.extend_from_slice(self.cube.container.indices().list());
        }

        for key in removable_stratums.iter() {
            self.strata.remove(key);
        }
    }
    fn flight_on_full(&mut self) {
        unsafe {
            self.cube.adjusted_data.resize(
                self.balancing_data.nrow(),
                self.balancing_data.ncol() + self.strata.len(),
            );
            self.cube.candidate_data.resize(
                self.cube.adjusted_data.ncol(),
                self.cube.adjusted_data.ncol() + 1,
            );
        }

        let mut all_units = Vec::<usize>::new();

        for (si, (_, stratum)) in self.strata.iter().enumerate() {
            all_units.extend_from_slice(stratum);

            for &id in stratum.iter() {
                self.cube.adjusted_data[(id, si + self.balancing_data.ncol())] = 1.0;
            }
        }

        self.cube.variant.reset_to(
            &mut self.cube.container,
            &mut all_units,
            self.data,
            self.cube.adjusted_data.ncol() + 1,
        );

        self.cube.run_flight();

        // Fix stratas
        self.strata.clear();

        for &id in self.cube.container.indices().list().iter() {
            let stratum = self.strata_vec[id];
            match self.strata.get_mut(&stratum) {
                Some(uvec) => {
                    uvec.push(id);
                }
                None => {
                    self.strata.insert(stratum, vec![id]);
                }
            };
        }
    }
    fn landing_per_stratum(&mut self) {
        unsafe {
            self.cube
                .adjusted_data
                .resize(self.balancing_data.nrow(), self.balancing_data.ncol() + 1);
            self.cube.candidate_data.resize(
                self.cube.adjusted_data.ncol(),
                self.cube.adjusted_data.ncol() + 1,
            );
        }

        for (_key, stratum) in self.strata.iter_mut() {
            for &id in stratum.iter() {
                self.cube.adjusted_data[(id, 0)] = 1.0;
                for j in 0..self.balancing_data.ncol() {
                    self.cube.adjusted_data[(id, j + 1)] =
                        self.balancing_data[(id, j)] / self.probabilities[id];
                }
            }

            self.cube.variant.reset_to(
                &mut self.cube.container,
                stratum,
                self.data,
                self.cube.adjusted_data.ncol() + 1,
            );

            self.cube.run_landing();
        }
    }
}

pub fn cube_stratified<'a, R>(
    rand: &'a mut R,
    probabilities: &[f64],
    eps: f64,
    balancing_data: &'a RefMatrix,
    strata: &[i64],
) -> Vec<usize>
where
    R: RandomGenerator,
{
    assert_eq!(strata.len(), probabilities.len());
    assert_eq!(balancing_data.nrow(), probabilities.len());

    let mut units = HashMap::<i64, Vec<usize>, FxSeededState>::with_capacity_and_hasher(
        probabilities.len() / 10,
        FxSeededState::with_seed(rand.rusize(probabilities.len())),
    );
    let mut amat = Box::new(Matrix::new_fill(
        0.0,
        (balancing_data.nrow(), balancing_data.ncol() + 1),
    ));
    let candidate_dim = (amat.ncol(), amat.ncol() + 1);
    let container = Box::new(Container::new(rand, probabilities, eps));

    for i in 0..probabilities.len() {
        if !container.indices().contains(i) {
            continue;
        }

        let stratum = strata[i];
        match units.get_mut(&stratum) {
            Some(uvec) => {
                uvec.push(i);
            }
            None => {
                units.insert(stratum, vec![i]);
            }
        };

        amat[(i, 0)] = 1.0;
        for j in 0..balancing_data.ncol() {
            amat[(i, j + 1)] = balancing_data[(i, j)] / probabilities[i];
        }
    }

    let mut cube = CubeMethodSampler {
        container: container,
        variant: Box::new(CubeMethod {}),
        candidates: Vec::<usize>::with_capacity(20),
        adjusted_data: amat,
        candidate_data: Box::new(Matrix::new_fill(0.0, candidate_dim)),
    };

    // Flight per stratum
    let mut removable_stratums = Vec::<i64>::new();
    for (stratum_key, stratum) in units.iter_mut() {
        cube.container.indices_mut().clear();
        for unit in stratum.iter() {
            cube.container.indices_mut().insert(*unit);
        }

        cube.run_flight();

        if cube.container.indices().len() == 0 {
            removable_stratums.push(*stratum_key);
            continue;
        }

        stratum.clear();
        stratum.extend_from_slice(cube.container.indices().list());
    }

    for key in removable_stratums.iter() {
        units.remove(key);
    }

    if units.len() == 0 {
        return cube.get_sorted_sample().to_vec();
    }

    // Flight on full
    cube.container.indices_mut().clear();
    unsafe {
        cube.adjusted_data
            .resize(balancing_data.nrow(), balancing_data.ncol() + units.len());
        cube.candidate_data
            .resize(cube.adjusted_data.ncol(), cube.adjusted_data.ncol() + 1);
    }

    for (si, (_, stratum)) in units.iter().enumerate() {
        for &unit in stratum.iter() {
            cube.container.indices_mut().insert(unit);
            cube.adjusted_data[(unit, si)] = 1.0;
            for j in 0..balancing_data.ncol() {
                cube.adjusted_data[(unit, j + units.len())] =
                    balancing_data[(unit, j)] / probabilities[unit];
            }
        }
    }

    cube.run_flight();

    if cube.container.indices().len() == 0 {
        return cube.get_sorted_sample().to_vec();
    }

    // Landing per stratum
    units.clear();
    unsafe {
        cube.adjusted_data
            .resize(balancing_data.nrow(), balancing_data.ncol() + 1);
        cube.candidate_data
            .resize(cube.adjusted_data.ncol(), cube.adjusted_data.ncol() + 1);
    }

    for &id in cube.container.indices().list().iter() {
        let stratum = strata[id];
        match units.get_mut(&stratum) {
            Some(uvec) => {
                uvec.push(id);
            }
            None => {
                units.insert(stratum, vec![id]);
            }
        };

        cube.adjusted_data[(id, 0)] = 1.0;
        for j in 0..balancing_data.ncol() {
            cube.adjusted_data[(id, j + 1)] = balancing_data[(id, j)] / probabilities[id];
        }
    }

    for (_, stratum) in units.iter_mut() {
        cube.container.indices_mut().clear();
        for unit in stratum.iter() {
            cube.container.indices_mut().insert(*unit);
        }

        cube.run_landing();
    }

    cube.get_sorted_sample().to_vec()
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
    use crate::test_utils::{assert_delta, data_10_2, data_20_2, gen_rand, TestRandom, EPS};

    #[test]
    fn cube() {
        let balmat = RefMatrix::new(
            &[
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, //
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            ],
            10,
        );
        let (mut rand00, mut _rand99) = gen_rand();
        let (_sprmat, prob) = data_10_2();

        let mut cube = CubeMethod::new(&mut rand00, &prob, EPS, &balmat);
        cube.run_flight();
        assert!(cube.container.indices().len() < 3);

        let s = CubeMethod::sample(&mut rand00, &prob, EPS, &balmat);
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
        let (mut rand00, mut _rand99) = gen_rand();
        let (sprmat, prob) = data_10_2();

        let mut cube = LocalCubeMethod::new(&mut rand00, &prob, EPS, &balmat, &sprmat, 2);
        cube.run_flight();
        assert!(cube.container.indices().len() < 3);

        let s = LocalCubeMethod::sample(&mut rand00, &prob, EPS, &balmat, &sprmat, 2);
        assert!(s.len() == 2);
    }

    #[test]
    fn cube_stratified() {
        let (mut rand00, mut _rand99) = gen_rand();

        {
            let (balmat, prob) = data_10_2();
            let s = CubeMethod::sample_stratified(
                &mut rand00,
                &prob,
                EPS,
                &balmat,
                &[1i64, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            );
            assert_eq!(s.len(), 2);
            assert!((0..5).contains(&s[0]));
            assert!((5..10).contains(&s[1]));
        }
        {
            let (balmat, prob) = data_20_2();
            let s = CubeMethod::sample_stratified(
                &mut rand00,
                &prob,
                EPS,
                &balmat,
                &[
                    1i64, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                ],
            );
            assert_eq!(s.len(), 4);
            assert!((0..5).contains(&s[0]));
            assert!((5..10).contains(&s[1]));
            assert!((10..20).contains(&s[2]));
            assert!((10..20).contains(&s[3]));
        }
    }

    #[test]
    fn lcube_stratified() {
        let mut rand = TestRandom::new(0);

        {
            let (balmat, prob) = data_10_2();
            let s = LocalCubeMethod::sample_stratified(
                &mut rand,
                &prob,
                EPS,
                &balmat,
                &balmat,
                2,
                &[1i64, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            );
            assert_eq!(s.len(), 2);
            assert!((0..5).contains(&s[0]));
            assert!((5..10).contains(&s[1]));
        }
        {
            let (balmat, prob) = data_20_2();

            for _ in 0..4 {
                let s = LocalCubeMethod::sample_stratified(
                    &mut rand,
                    &prob,
                    EPS,
                    &balmat,
                    &balmat,
                    2,
                    &[
                        1i64, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    ],
                );
                assert_eq!(s.len(), 4);
                assert!((0..5).contains(&s[0]));
                assert!((5..10).contains(&s[1]));
                assert!((10..20).contains(&s[2]));
                assert!((10..20).contains(&s[3]));
            }
        }
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
