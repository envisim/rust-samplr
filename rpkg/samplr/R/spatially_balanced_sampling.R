#' Spatially balanced sampling
#'
#' @name Spatially balanced sampling
#' @rdname spatially_balanced_sampling
#' @description
#' Selects spatially balanced samples with prescribed inclusion probabilities from finite
#' populations.
#'
#' @param probabilities A vector of inclusion probabilities.
#' @param spread_mat A matrix of spreading covariates.
#' @inheritDotParams .sampling_defaults -max_iter
#'
#' @references
#' Deville, J.-C., &  Tillé, Y. (1998).
#' Unequal probability sampling without replacement through a splitting method.
#' Biometrika 85, 89-101.
#'
#' Grafström, A. (2012).
#' Spatially correlated Poisson sampling.
#' Journal of Statistical Planning and Inference, 142(1), 139-147.
#'
#' Grafström, A., Lundström, N.L.P. & Schelin, L. (2012).
#' Spatially balanced sampling through the Pivotal method.
#' Biometrics 68(2), 514-520.
#'
#' Lisic, J. J., & Cruze, N. B. (2016, June).
#' Local pivotal methods for large surveys.
#' In Proceedings of the Fifth International Conference on Establishment Surveys.
#'
#' Prentius, W. (2024).
#' Locally correlated Poisson sampling.
#' Environmetrics, 35(2), e2832.
#'
#' @examples
#' \dontrun{
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' x = matrix(runif(N * 2), ncol = 2);
#' s = lpm_2(prob, x);
#' plot(x[, 1], x[, 2]);
#' points(x[s, 1], x[s, 2], pch = 19);
#'
#' set.seed(12345);
#' prob = c(0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9);
#' N = length(prob);
#' x = matrix(runif(N * 2), ncol = 2);
#' ep = rep(0L, N);
#' r = 10000L;
#' for (i in seq_len(r)) {
#'   s = lpm_2(prob, x);
#'   ep[s] = ep[s] + 1L;
#' }
#' print(ep / r);
#'
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' x = matrix(runif(N * 2), ncol = 2);
#' lpm_1(prob, x);
#' lpm_2(prob, x);
#' lpm_1s(prob, x);
#' scps(prob, x);
#' lcps(prob, x);
#'
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' x = matrix(runif(N * 2), ncol = 2);
#' sizes = c(10, 20, 30, 40);
#' s = lpm_2_hierarchical(prob, x, sizes);
#' plot(x[, 1], x[, 2]);
#' points(x[s, 1], x[s, 2], pch = 19);
#' }
#'
NULL

.spatially_balanced_wrapper = function(method, probabilities, spread_mat, ...) {
  args = .sampling_defaults(...);
  .Call(
    wrap__rust_spatially_balanced,
    probabilities,
    spread_mat,
    args$eps,
    args$bucket_size,
    args$seed,
    method
  ) + 1L
}

#' @describeIn spatially_balanced_sampling Local pivotal method 1
lpm_1 = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("lpm_1", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Local pivotal method 1s
lpm_1s = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("lpm_1s", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Local pivotal method 2
lpm_2 = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("lpm_2", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Spatially correlated Poisson sampling
scps = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("scps", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Locally correlated Poisson sampling
lcps = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("lcps", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Hierarchical Local pivotal method 2
lpm_2_hierarchical = function(probabilities, spread_mat, sizes, ...) {
  args = .sampling_defaults(...);
  .Call(
    wrap__rust_spatially_balanced_hierarchical,
    probabilities,
    spread_mat,
    sizes,
    args$eps,
    args$bucket_size,
    args$seed,
    "lpm_2"
  );
}
