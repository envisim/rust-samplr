#' Variance estimator for spatially balanced samples
#'
#' @description
#' Variance estimator of HT estimator of population total.
#'
#' @param values A vector of values of the variable of interest.
#' @param probabilities A vector of inclusion probabilities.
#' @param spread_mat A matrix of spreading covariates.
#' @param n_neighbours The number of neighbours to construct the means around.
#'
#' @references
#' Grafström, A., & Schelin, L. (2014).
#' How to select representative samples.
#' Scandinavian Journal of Statistics, 41(2), 277-290.
#'
#' @examples
#' \dontrun{
#' set.seed(12345);
#' N = 1000;
#' n = 100;
#' prob = rep(n/N, N);
#' xs = matrix(runif(N * 2), ncol = 2);
#' y = runif(N);
#'
#' s = lpm_2(prob, xs);
#' local_mean_variance(y[s], prob[s], xs[s, ], 4);
#' BalancedSampling::vsb(prob[s], y[s], xs[s, ], 4);
#'
#' // Compare SRS, empirical
#' N = 1000;
#' n = 20;
#' prob = rep(n/N, N);
#' r = 1000L;
#' v = matrix(0.0, r, 3L+1);
#'
#' for (i in seq_len(r)) {
#'   s = lpm_2(prob, xs);
#'   v[i, 1] = local_mean_variance(y[s], prob[s], xs[s, ], 4);
#'   v[i, 2] = N^2 * sd(y[s]) / n;
#'   v[i, 3] = sum(y[s] / prob[s]);
#'   v[i, 4] = BalancedSampling::vsb(prob[s], y[s], xs[s, ], 4);
#' }
#'
#' print(c(mean(v[, 1]), mean(v[, 2]), var(v[, 3])));
#' }
#'
#' @export
local_mean_variance = function(values, probabilities, spread_mat, neighbours = 4L) {
  neighbours = as.integer(neighbours);
  if (neighbours < 1L) {
    stop("neighbours needs to be positive");
  } else if (neighbours == 1L) {
    return (NaN);
  }

  rust_local_mean_variance(
    as.double(values),
    as.double(probabilities),
    as.matrix(spread_mat),
    neighbours
  )
}
