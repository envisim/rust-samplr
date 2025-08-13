#' Spatial balance measure
#'
#' @name Spatial balance measure
#' @rdname spatial_balance_measure
#' @description
#' Calculates the spatial balance of a sample.
#'
#' @param sample A vector of sample indices.
#' @param probabilities A vector of inclusion probabilities.
#' @param spread_mat A matrix of spreading covariates.
#'
#' @returns the measure, or in case of `balance_deviation`, the vector of deviations
#'
#' @references
#' Stevens Jr, D. L., & Olsen, A. R. (2004).
#' Spatially balanced sampling of natural resources.
#' Journal of the American statistical Association, 99(465), 262-278.
#'
#' Grafström, A., Lundström, N.L.P. & Schelin, L. (2012).
#' Spatially balanced sampling through the Pivotal method.
#' Biometrics 68(2), 514-520.
#'
#' Prentius, W., & Grafström, A. (2024).
#' How to find the best sampling design: A new measure of spatial balance.
#' Environmetrics, 35(7), e2878.
#'
#' @examples
#' \dontrun{
#' set.seed(12345);
#' N = 500;
#' n = 70;
#' prob = rep(n / N, N);
#' maux = matrix(runif(N * 2), ncol = 2);
#' s = lpm_2(prob, maux);
#' sb_v = spatial_balance_voronoi(s, prob, maux);
#' sb_l = spatial_balance_local(s, prob, maux);
#' }
#'
NULL

.spatial_balance_wrapper = function(method, sample, probabilities, spread_mat) {
  .Call(
    wrap__rust_spatial_balance_measure,
    sample - 1L,
    probabilities,
    spread_mat,
    method
  )
}

#' @describeIn spatial_balance_measure Local spatial balance
spatial_balance_local = function(sample, probabilities, spread_mat) {
  .spatial_balance_wrapper("local", sample, probabilities, spread_mat)
}

#' @describeIn spatial_balance_measure Voronoi spatial balance
spatial_balance_voronoi = function(sample, probabilities, spread_mat) {
  .spatial_balance_wrapper("voronoi", sample, probabilities, spread_mat)
}

#' @describeIn spatial_balance_measure Balance deviation
balance_devaition = function(sample, probabilities, spread_mat) {
  .Call(
    wrap__rust_balance_deviation,
    sample - 1L,
    probabilities,
    spread_mat,
  )
}
