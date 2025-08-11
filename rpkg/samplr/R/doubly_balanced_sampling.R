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
