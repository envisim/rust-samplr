#' Balanced sampling
#'
#' @name Balanced sampling
#' @rdname balanced_sampling
#' @description
#' Selects balanced samples with prescribed inclusion probabilities from finite populations.
#'
#' @details
#' For the `cube` method, a fixed sized sample is obtained if the first column of `balance_mat` is
#' the inclusion probabilities. For `cube_stratified`, the inclusion probabilities are inserted
#' automatically.
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
#' xb = matrix(c(prob, runif(N * 2)), ncol = 3);
#' strata = c(rep(1L, 100), rep(2L, 200), rep(3L, 300), rep(4L, 400));
#'
#' s = cube(prob, xb);
#' plot(xb[, 2], xb[, 3], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = cube_stratified(prob, xb[, -1], strata);
#' plot(xb[, 2], xb[, 3], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' # Respects inclusion probabilities
#' set.seed(12345);
#' prob = c(0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9);
#' N = length(prob);
#' xb = matrix(c(prob, runif(N * 2)), ncol = 3);
#'
#' ep = rep(0L, N);
#' r = 10000L;
#'
#' for (i in seq_len(r)) {
#'   s = cube(prob, xb);
#'   ep[s] = ep[s] + 1L;
#' }
#'
#' print(ep / r - prob);
#' }
#'
NULL

.balanced_wrapper = function(method, probabilities, balance_mat, ...) {
  args = .sampling_defaults(...);
  rust_balanced(
    as.double(probabilities),
    as.matrix(balance_mat),
    args$eps,
    args$seed,
    method
  )
}

#' @describeIn balanced_sampling The cube method
#' @export
cube = function(probabilities, balance_mat, ...) {
  .balanced_wrapper("cube", probabilities, balance_mat, ...)
}

#' @describeIn balanced_sampling The stratified cube method
#' @export
cube_stratified = function(probabilities, balance_mat, strata, ...) {
  args = .sampling_defaults(...);
  rust_balanced_stratified(
    as.numeric(probabilities),
    as.matrix(balance_mat),
    as.integer(strata),
    args$eps,
    args$seed,
    "cube"
  )
}
