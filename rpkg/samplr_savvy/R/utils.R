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
    eps = as.double(eps),
    seed = as.integer(seed),
    max_iter = as.integer(max_iter),
    bucket_size = as.integer(bucket_size)
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

  args
}

#' Transform a sample vector into an inclusion indicator vector
#'
#' @param sample A vector of sample indices.
#'
#' @export
sample_to_indicator = function(sample, population_size) as.integer(seq_len(population_size) %in% sample)
