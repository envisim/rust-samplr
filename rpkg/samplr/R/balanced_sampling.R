#' Balanced sampling
#'
#' @name Balanced sampling
#' @rdname balanced_sampling
#' @description
#' Selects balanced samples with prescribed inclusion probabilities from finite populations.
#'
#' @details
#' For the cube method, a fixed sized sample is obtained if the first column of `balance_mat` is the
#' inclusion probabilities.
#'
#' @param probabilities A vector of inclusion probabilities.
#' @param balance_mat A matrix of balancing covariates.
#' @inheritDotParams .sampling_defaults -max_iter -bucket_size
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
#' @examples
#' \dontrun{
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' x = matrix(runif(N * 2), ncol = 2);
#' s = cube(prob, x);
#' plot(x[, 1], x[, 2]);
#' points(x[s, 1], x[s, 2], pch = 19);
#'
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' x = matrix(runif(N * 2), ncol = 2);
#' strata = c(rep(1L, 100), rep(2L, 200), rep(3L, 300), rep(4L, 400));
#' s = cube_stratified(prob, x, strata);
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
#'   s = cube(prob, cbind(prob, x));
#'   ep[s] = ep[s] + 1L;
#' }
#' print(ep / r);
#' }
#'
NULL

.balanced_wrapper = function(method, probabilities, balance_mat, ...) {
  args = .sampling_defaults(...);
  .Call(
    wrap__rust_balanced,
    probabilities,
    balance_mat,
    args$eps,
    args$seed,
    method
  );
}

#' @describeIn balanced_sampling The cube method
cube = function(probabilities, balance_mat, ...) {
  .balanced_wrapper("cube", probabilities, balance_mat, ...)
}

#' @describeIn balanced_sampling The stratified cube method
cube_stratified = function(probabilities, balance_mat, strata, ...) {
  args = .sampling_defaults(...);
  .Call(
    wrap__rust_balanced_stratified,
    probabilities,
    balance_mat,
    strata,
    args$eps,
    args$seed,
    "cube"
  );
}
