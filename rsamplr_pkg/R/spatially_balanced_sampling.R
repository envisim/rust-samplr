#' Spatially balanced sampling
#'
#' @name Spatially balanced sampling
#' @rdname spatially_balanced_sampling
#' @description
#' Selects spatially balanced samples with prescribed inclusion probabilities from finite
#' populations.
#'
#' @details
#' `lpm_2_hierarchical` selects an initial sample using the LPM2 algorithm, and then splits this
#' sample into subsamples of given `sizes`, using successive, hierarchical selection with LPM2.
#' When using `lpm_2_hierarchical`, the inclusion probabilities must sum to an integer, and the
#' `sizes` vector (the subsamples) must sum to the same integer.
#'
#' @param probabilities A vector of inclusion probabilities.
#' @param spread_mat A matrix of spreading covariates.
#' @param sizes A vector of integers containing the sizes of the subsamples.
#' @inheritDotParams .sampling_defaults -max_iter
#'
#' @returns A vector of sample indices, or in the case of hierarchical sampling, a matrix where the
#' first column contains sample indices and the second column contains subsample indices (groups).
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
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' xs = matrix(runif(N * 2), ncol = 2);
#' sizes = c(10L, 20L, 30L, 40L);
#'
#' s = lpm_1(prob, xs);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = lpm_1s(prob, xs);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = lpm_2(prob, xs);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = scps(prob, xs);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = lpm_2_hierarchical(prob, xs, sizes);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' \donttest{
#' s = lcps(prob, xs); # May have a long execution time
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' # Respects inclusion probabilities
#' set.seed(12345);
#' prob = c(0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9);
#' N = length(prob);
#' xs = matrix(c(prob, runif(N * 2)), ncol = 3);
#'
#' ep = rep(0L, N);
#' r = 10000L;
#'
#' for (i in seq_len(r)) {
#'   s = lpm_2(prob, xs);
#'   ep[s] = ep[s] + 1L;
#' }
#'
#' print(ep / r - prob);
#' }
#'
NULL

.spatially_balanced_wrapper = function(method, probabilities, spread_mat, ...) {
  args = .sampling_defaults(...);

  rust_spatially_balanced(
    as.double(probabilities),
    as.matrix(spread_mat),
    args$eps,
    args$bucket_size,
    method
  )
}

#' @describeIn spatially_balanced_sampling Local pivotal method 1
#' @export
lpm_1 = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("lpm_1", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Local pivotal method 1s
#' @export
lpm_1s = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("lpm_1s", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Local pivotal method 2
#' @export
lpm_2 = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("lpm_2", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Spatially correlated Poisson sampling
#' @export
scps = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("scps", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Locally correlated Poisson sampling
#' @export
lcps = function(probabilities, spread_mat, ...) {
  .spatially_balanced_wrapper("lcps", probabilities, spread_mat, ...)
}

#' @describeIn spatially_balanced_sampling Hierarchical Local pivotal method 2
#' @export
lpm_2_hierarchical = function(probabilities, spread_mat, sizes, ...) {
  args = .sampling_defaults(...);
  rust_spatially_balanced_hierarchical(
    as.double(probabilities),
    as.matrix(spread_mat),
    as.integer(sizes),
    args$eps,
    args$bucket_size,
    "lpm_2"
  )
}
