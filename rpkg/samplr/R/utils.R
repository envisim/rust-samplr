#' Sampling defaults
#'
#' @param eps A small value used when comparing floats.
#' @param seed An integer used to determine the seed for the random number generator.
#' @param max_iter The maximum number of iterations used in iterative algorithms.
#' @param bucket_size The maximum size of the k-d-tree nodes. A higher value gives a slower
#' k-d-tree, but is faster to create and takes up less memory.
#' @keywords internal
.sampling_defaults = function(
  eps = 1e-12,
  seed = sample.int(.Machine$integer.max, 1L),
  max_iter = 1000L,
  bucket_size = 50L
) {
  args = list(
    as.numeric(eps),
    as.integer(seed),
    as.integer(max_iter),
    as.integer(bucket_size)
  );

  if (!(length(args$eps) == 1 && 0.0 <= args$eps && args$eps <= 1e-3)) {
    args$eps = 1e-12;
  }

  if (!(length(args$bucket_size) == 1 && 0 < args$bucket_size)) {
    args$bucket_size = 50L;
  }

  if (!(length(args$max_iter) == 1 && 0 < args$max_iter)) {
    args$max_iter = 1000L;
  }

  if (!(length(args$seed) == 1 && 0 < args$seed)) {
    args$seed = sample.int(.Machine$integer.max, 1L);
  }


  ## arguments = list(...);
  ## rargs = list(
  ##   eps = 1e-12,
  ##   seed = 0L,
  ##   max_iter = 1000L,
  ##   bucket_size = 50L
  ## );

  ## eps = arguments["eps"];
  ## if (!is.null(eps) && is.numeric(eps) && length(arguments$eps) == 1 && 0.0 <= eps && eps <= 1e-3) {
  ##   rargs$eps = eps;
  ## }

  ## bucket_size = arguments["bucket_size"];
  ## if (!is.null(bucket_size) && is.numeric(bucket_size) && length(bucket_size) == 1 && 0 < bucket_size) {
  ##   rargs$bucket_size = as.integer(bucket_size);
  ## }

  ## max_iter = arguments["max_iter"];
  ## if (!is.null(max_iter) && is.numeric(max_iter) && length(max_iter) == 1 && 0 < max_iter) {
  ##   rargs$max_iter = as.integer(max_iter);
  ## }

  ## seed = arguments["seed"];
  ## if (!is.null(seed) && is.numeric(seed) && length(seed) == 1 && 0 < seed) {
  ##   rargs$seed = as.integer(seed);
  ## } else {
  ##   rargs$seed = sample.int(.Machine$integer.max, 1L);
  ## }

  args
}

