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
