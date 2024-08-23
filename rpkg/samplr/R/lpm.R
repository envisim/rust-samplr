#' The (Local) Pivotal Method 2 (LPM2)
#'
#' @export
lpm_2 = function(
  prob,
  spr_aux,
  eps = 1e-12,
  bucket_size = 50,
  seed = sample.int(.Machine$integer.max, 1L)) {
  rust_lpm_2(prob, spr_aux, eps, bucket_size, seed)
}

