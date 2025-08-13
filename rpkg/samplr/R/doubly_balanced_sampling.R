#' Doubly balanced sampling
#'
#' @name Doubly balanced sampling
#' @rdname doubly_balanced_sampling
#' @description
#' Selects doubly balanced samples with prescribed inclusion probabilities from finite populations.
#'
#' @param probabilities A vector of inclusion probabilities.
#' @param spread_mat A matrix of spreading covariates.
#' @param balance_mat A matrix of balancing covariates.
#' @inheritDotParams .sampling_defaults -max_iter
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
#' \dontrun{
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' x = matrix(runif(N * 2), ncol = 2);
#' xspr = matrix(runif(N * 2), ncol = 2);
#' s = local_cube(prob, xspr, cbind(prob, x));
#' plot(x[, 1], x[, 2]);
#' points(x[s, 1], x[s, 2], pch = 19);
#'
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' x = matrix(runif(N * 2), ncol = 2);
#' xspr = matrix(runif(N * 2), ncol = 2);
#' strata = c(rep(1L, 100), rep(2L, 200), rep(3L, 300), rep(4L, 400));
#' s = local_cube_stratified(prob, xspr, x, strata);
#' plot(x[, 1], x[, 2]);
#' points(x[s, 1], x[s, 2], pch = 19);
#'
#' set.seed(12345);
#' prob = c(0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9);
#' N = length(prob);
#' x = matrix(runif(N * 2), ncol = 2);
#' xspr = matrix(runif(N * 2), ncol = 2);
#' ep = rep(0L, N);
#' r = 10000L;
#' for (i in seq_len(r)) {
#'   s = local_cube(prob, xspr, cbind(prob, x));
#'   ep[s] = ep[s] + 1L;
#' }
#' print(ep / r);
#' }
#'
NULL

.balanced_wrapper = function(method, probabilities, spread_mat, balance_mat, ...) {
  args = .sampling_defaults(...);
  .Call(
    wrap__rust_doubly_balanced,
    probabilities,
    spread_mat,
    balance_mat,
    args$eps,
    args$seed,
    method
  ) + 1L
}

#' @describeIn balanced_sampling The local cube method
local_cube = function(probabilities, balance_mat, ...) {
  .balanced_wrapper("local_cube", probabilities, balance_mat, ...)
}

#' @describeIn balanced_sampling The stratified local cube method
local_cube_stratified = function(probabilities, spread_mat, balance_mat, strata, ...) {
  args = .sampling_defaults(...);
  .Call(
    wrap__rust_doubly_balanced_stratified,
    probabilities,
    spread_mat,
    balance_mat,
    strata,
    args$eps,
    args$seed,
    "local_cube"
  );
}
