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
#' x = matrix(runif(N * 2), ncol = 2);
#' y = runif(N);
#' s = lpm_2(prob, x);
#' local_mean_variance(y[s], prob[s], x[s, ]);
#' }
#'
local_mean_variance = function(values, probabilities, spread_mat, neighbours = 4L) {
  neighbours = as.integer(neighbours);
  if (neighbours < 1L) {
    stop("neighbours needs to be positive");
  } else if (neighbours == 1L) {
    return (NaN);
  }

  .Call(
    wrap__rust_local_mean_variance,
    values,
    probabilities,
    spread_mat,
    neighbours
  )
}
