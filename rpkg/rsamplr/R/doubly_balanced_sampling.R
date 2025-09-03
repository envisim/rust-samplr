#' Doubly balanced sampling
#'
#' @name Doubly balanced sampling
#' @rdname doubly_balanced_sampling
#' @description
#' Selects doubly balanced samples with prescribed inclusion probabilities from finite populations.
#'
#' @details
#' For the `local_cube` method, a fixed sized sample is obtained if the first column of
#' `balance_mat` is the inclusion probabilities. For `local_cube_stratified`, the inclusion
#' probabilities are inserted automatically.
#'
#' @param probabilities A vector of inclusion probabilities.
#' @param spread_mat A matrix of spreading covariates.
#' @param balance_mat A matrix of balancing covariates.
#' @param strata An integer vector with stratum numbers for each unit.
#' @inheritDotParams .sampling_defaults -max_iter
#'
#' @returns A vector of sample indices.
#'
#' @references
#' Deville, J. C. and Tillé, Y. (2004).
#' Efficient balanced sampling: the cube method.
#' Biometrika, 91(4), 893-912.
#'
#' Chauvet, G. and Tillé, Y. (2006).
#' A fast algorithm for balanced sampling.
#' Computational Statistics, 21(1), 53-62.
#'
#' Chauvet, G. (2009).
#' Stratified balanced sampling.
#' Survey Methodology, 35, 115-119.
#'
#' Grafström, A. and Tillé, Y. (2013).
#' Doubly balanced spatial sampling with spreading and restitution of auxiliary totals.
#' Environmetrics, 24(2), 120-131
#'
#' @examples
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' xb = matrix(c(prob, runif(N * 2)), ncol = 3);
#' xs = matrix(runif(N * 2), ncol = 2);
#'
#' s = local_cube(prob, xs, xb);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = local_cube_stratified(prob, xs, xb[, -1], strata);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' \donttest{
#' # Respects inclusion probabilities
#' set.seed(12345);
#' prob = c(0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9);
#' N = length(prob);
#' xb = matrix(c(prob, runif(N * 2)), ncol = 3);
#' xs = matrix(runif(N * 2), ncol = 2);
#'
#' ep = rep(0L, N);
#' r = 10000L;
#'
#' for (i in seq_len(r)) {
#'   s = local_cube(prob, xs, xb);
#'   ep[s] = ep[s] + 1L;
#' }
#'
#' print(ep / r - prob);
#' }
#'
NULL

.doubly_balanced_wrapper = function(method, probabilities, spread_mat, balance_mat, ...) {
  args = .sampling_defaults(...);
  rust_doubly_balanced(
    as.double(probabilities),
    as.matrix(spread_mat),
    as.matrix(balance_mat),
    args$eps,
    args$bucket_size,
    method
  )
}

#' @describeIn doubly_balanced_sampling The local cube method
#' @export
local_cube = function(probabilities, spread_mat, balance_mat, ...) {
  .doubly_balanced_wrapper("local_cube", probabilities, spread_mat, balance_mat, ...)
}

#' @describeIn doubly_balanced_sampling The stratified local cube method
#' @export
local_cube_stratified = function(probabilities, spread_mat, balance_mat, strata, ...) {
  args = .sampling_defaults(...);
  rust_doubly_balanced_stratified(
    as.double(probabilities),
    as.matrix(spread_mat),
    as.matrix(balance_mat),
    as.integer(strata),
    args$eps,
    args$bucket_size,
    "local_cube"
  );
}
