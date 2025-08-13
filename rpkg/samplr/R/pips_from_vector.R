#' Inclusion probabilities proportional-to-size
#'
#' @description
#' Computes the first-order inclusion probabilties from a vector of positive numbers,
#' for a probabilitiy proportional-to-size design.
#'
#' @param values A vector of positive numbers
#' @param sample_size The wanted sample size
#'
#' @return A vector of inclusion probabilities proportional-to-size
#'
#' @examples
#' \dontrun{
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' x = matrix(runif(N * 2), ncol = 2);
#' prob = pips_from_vector(x[, 1], n);
#' s = lpm_2(prob, x);
#' plot(x[, 1], x[, 2]);
#' points(x[s, 1], x[s, 2], pch = 19);
#' }
#'
pips_from_vector = function(values, sample_size) {
  values = as.numeric(values);
  N = length(values);
  sample_size = as.integer(sample_size);

  if (length(sample_size) != 1 || sample_size > N || sample_size < 0)
    stop("n must be integer in [0, N]");

  if (sample_size == N)
    return(rep(1.0, N));
  if (sample_size == 0)
    return(rep(0.0, N));

  .Call(
    wrap_rust_pips_from_values,
    values,
    sample_size
  )
}
