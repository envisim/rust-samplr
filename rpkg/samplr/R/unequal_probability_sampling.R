#' Unequal probability sampling
#'
#' @name Unequal probability sampling
#' @rdname unequal_probability_sampling
#' @description
#' Selects samples with prescribed inclusion probabilities from finite populations.
#'
#' @param probabilities A vector of inclusion probabilities.
#' @param sample_size The wanted sample size
#' @inheritDotParams .sampling_defaults -bucket_size
#'
#' @references
#' Bondesson, L., & Thorburn, D. (2008).
#' A list sequential sampling method suitable for real‐time sampling.
#' Scandinavian Journal of Statistics, 35(3), 466-483.
#'
#' Brewer, K. E. (1975).
#' A Simple Procedure for Sampling πpswor1.
#' Australian Journal of Statistics, 17(3), 166-172.
#'
#' Chauvet, G. (2012).
#' On a characterization of ordered pivotal sampling.
#' Bernoulli, 18(4), 1320-1340.
#'
#' Deville, J.-C., &  Tillé, Y. (1998).
#' Unequal probability sampling without replacement through a splitting method.
#' Biometrika 85, 89-101.
#'
#' Grafström, A. (2009).
#' Non-rejective implementations of the Sampford sampling design.
#' Journal of Statistical Planning and Inference, 139(6), 2111-2114.
#'
#' Rosén, B. (1997).
#' On sampling with probability proportional to size.
#' Journal of statistical planning and inference, 62(2), 159-191.
#'
#' Sampford, M. R. (1967).
#' On sampling without replacement with unequal probabilities of selection.
#' Biometrika, 54(3-4), 499-513.
#'
#' @examples
#' \dontrun{
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' rpm(prob);
#' spm(prob);
#' cps(prob);
#' poisson(prob);
#' conditional_poisson(prob, n);
#' brewer(prob);
#' pareto(prob);
#' sampford(prob);
#' }
#'
NULL

.unequal_wrapper = function(method, probabilities, ...) {
  args = .sampling_defaults(...);
  .Call(
    wrap__rust_unequal,
    probabilities,
    args$eps,
    args$seed,
    method,
    args$max_iter
  ) + 1L
}

#' @describeIn unequal_probability_sampling Random pivotal method
rpm = function(probabilities, ...) {
  .unequal_wrapper("rpm", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Sequential pivotal method
spm = function(probabilities, ...) {
  .unequal_wrapper("spm", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Correlated Poisson sampling
cps = function(probabilities, ...) {
  .unequal_wrapper("cps", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Poisson sampling
poisson = function(probabilities, ...) {
  .unequal_wrapper("poisson", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Conditional Poisson sampling
conditional_poisson = function(probabilities, sample_size, ...) {
  args = .sampling_defaults(...);
  .Call(
    wrap__rust_unequal_conditional_poisson,
    probabilities,
    sample_size,
    args$eps,
    args$seed,
    args$max_iter
  ) + 1L
}

#' @describeIn unequal_probability_sampling Systematic sampling
systematic = function(probabilities, ...) {
  .unequal_wrapper("systematic", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Systematic sampling with random order
systematic_random_order = function(probabilities, ...) {
  .unequal_wrapper("systematic_random_order", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Brewer sampling
brewer = function(probabilities, ...) {
  .unequal_wrapper("brewer", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Pareto sampling
pareto = function(probabilities, ...) {
  .unequal_wrapper("pareto", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Sampford sampling
sampford = function(probabilities, ...) {
  .unequal_wrapper("sampford", probabilities, ...)
}
