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
#' A Simple Procedure for Sampling pi-ps wor.
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
#' xs = matrix(runif(N * 2), ncol = 2);
#'
#' s = rpm(prob);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = spm(prob);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = cps(prob);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = poisson(prob);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = conditional_poisson(prob, n);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = brewer(prob);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = pareto(prob);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' s = sampford(prob);
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' # Respects inclusion probabilities
#' set.seed(12345);
#' prob = c(0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9);
#' N = length(prob);
#'
#' ep = rep(0L, N);
#' r = 10000L;
#'
#' for (i in seq_len(r)) {
#'   s = poisson(prob);
#'   ep[s] = ep[s] + 1L;
#' }
#'
#' print(ep / r - prob);
#' }
#'
NULL

.unequal_wrapper = function(method, probabilities, ...) {
  args = .sampling_defaults(...);
  rust_unequal(
    as.double(probabilities),
    args$eps,
    args$seed,
    method,
    args$max_iter
  )
}

#' @describeIn unequal_probability_sampling Random pivotal method
#' @export
rpm = function(probabilities, ...) {
  .unequal_wrapper("rpm", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Sequential pivotal method
#' @export
spm = function(probabilities, ...) {
  .unequal_wrapper("spm", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Correlated Poisson sampling
#' @export
cps = function(probabilities, ...) {
  .unequal_wrapper("cps", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Poisson sampling
#' @export
poisson = function(probabilities, ...) {
  .unequal_wrapper("poisson", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Conditional Poisson sampling
#' @export
conditional_poisson = function(probabilities, sample_size, ...) {
  args = .sampling_defaults(...);
  rust_unequal_conditional_poisson(
    as.double(probabilities),
    as.integer(sample_size),
    args$eps,
    args$seed,
    args$max_iter
  )
}

#' @describeIn unequal_probability_sampling Systematic sampling
#' @export
systematic = function(probabilities, ...) {
  .unequal_wrapper("systematic", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Systematic sampling with random order
#' @export
systematic_random_order = function(probabilities, ...) {
  .unequal_wrapper("systematic_random_order", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Brewer sampling
#' @export
brewer = function(probabilities, ...) {
  .unequal_wrapper("brewer", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Pareto sampling
#' @export
pareto = function(probabilities, ...) {
  .unequal_wrapper("pareto", probabilities, ...)
}

#' @describeIn unequal_probability_sampling Sampford sampling
#' @export
sampford = function(probabilities, ...) {
  .unequal_wrapper("sampford", probabilities, ...)
}
