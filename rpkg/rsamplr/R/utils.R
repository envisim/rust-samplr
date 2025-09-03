#' Sampling defaults
#'
#' @param eps A small value used when comparing floats.
#' @param max_iter The maximum number of iterations used in iterative algorithms.
#' @param bucket_size The maximum size of the k-d-tree nodes. A higher value gives a slower
#' k-d-tree, but is faster to create and takes up less memory.
#' @keywords internal
.sampling_defaults = function(
  eps = 1e-10,
  max_iter = 1000L,
  bucket_size = 50L
) {
  args = list(
    eps = as.double(eps),
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

  args
}

#' Transform a sample vector into an inclusion indicator vector
#'
#' @param sample A vector of sample indices.
#' @param population_size The total size of the population.
#'
#' @returns An inclusion indicator vector, i.e. a `population_size`-sized vector of 0/1.
#'
#' @export
sample_to_indicator = function(sample, population_size) as.integer(seq_len(population_size) %in% sample)
