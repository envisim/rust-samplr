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
#' @returns the measure, or in case of `balance_deviation`, the vector of deviations.
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
#' set.seed(12345);
#' N = 500;
#' n = 70;
#' prob = rep(n / N, N);
#' xs = matrix(runif(N * 2), ncol = 2);
#'
#' s = lpm_2(prob, xs);
#' spatial_balance_voronoi(s, prob, xs);
#' spatial_balance_local(s, prob, xs);
#' balance_deviation(s, prob, xs);
#'
#' \donttest{
#' # Compare SRS
#' r = 1000L;
#' sb_v = matrix(0.0, r, 2L);
#' sb_l = matrix(0.0, r, 2L);
#' bal = matrix(0.0, r, 2L * ncol(xs));
#'
#' for (i in seq_len(r)) {
#'   s1 = lpm_2(prob, xs);
#'   s2 = sample(N, n);
#'   sb_v[i, ] = c(
#'     spatial_balance_voronoi(s1, prob, xs),
#'     spatial_balance_voronoi(s2, prob, xs)
#'   );
#'   sb_l[i, ] = c(
#'     spatial_balance_local(s1, prob, xs),
#'     spatial_balance_local(s2, prob, xs)
#'   );
#'   bal[i, ] = c(
#'     balance_deviation(s1, prob, xs),
#'     balance_deviation(s2, prob, xs)
#'   );
#' }
#'
#' # Spatial balance measure (voronoi), LPM vs SRS
#' print(colMeans(sb_v));
#' # Spatial balance measure (local), LPM vs SRS
#' print(colMeans(sb_l));
#' # Abs. balance deviation, LPM vs SRS
#' print(colMeans(abs(bal)));
#' }
#'
NULL

.spatial_balance_measure_wrapper = function(method, sample, probabilities, spread_mat) {
  rust_spatial_balance_measure(
    as.integer(sample),
    as.double(probabilities),
    as.matrix(spread_mat),
    method
  )
}

#' @describeIn spatial_balance_measure Local spatial balance
#' @export
spatial_balance_local = function(sample, probabilities, spread_mat) {
  .spatial_balance_measure_wrapper("local", sample, probabilities, spread_mat)
}

#' @describeIn spatial_balance_measure Voronoi spatial balance
#' @export
spatial_balance_voronoi = function(sample, probabilities, spread_mat) {
  .spatial_balance_measure_wrapper("voronoi", sample, probabilities, spread_mat)
}

#' @describeIn spatial_balance_measure Balance deviation
#' @export
balance_deviation = function(sample, probabilities, spread_mat) {
  rust_balance_deviation(
    as.integer(sample),
    as.double(probabilities),
    as.matrix(spread_mat)
  )
}
