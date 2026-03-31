#' Draw from a sampling design
#'
#' @param design A design object.
#' @param ... Additional arguments passed to methods.
#'
#' @describeIn draw.dbd draw generic
#' @export
draw = function(design, ...) {
  UseMethod("draw");
}

#' Draw a sample from a distributionally balanced design
#'
#' @param design A `dbd` design object.
#' @param ... Additional arguments passed to methods. `draw.dbd` accepts `sample_id` (integer). If
#' `sample_id` is `0L`, returns a random sample (default).
#'
#' @returns a vector of sample indices.
#'
#' @examples
#' set.seed(12345);
#' N = 1000L;
#' n = 100L;
#' prob = rep(n / N, N);
#' xs = matrix(runif(N * 2), ncol = 2);
#'
#' # Construct dbd design using tactical configuration
#' design = dbd_tc(
#'   n,
#'   xs,
#'   annealing_temp = 0.1,
#'   annealing_cooling = 0.999,
#'   max_iter = 1e5L
#' );
#' s = draw(design); # Draw sample from design
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#'
#' @export
draw.dbd = function(design, ...) {
  args = list(...);
  method = attr(design, "method");
  no_samples = attr(design, "number_of_samples");

  sample_id = as.integer(args$sample_id);
  if (length(sample_id) == 0 || !(0L < sample_id && sample_id <= no_samples)) {
    sample_id = sample(no_samples, 1L);
  }

  if (method == "dbd_circular") {
    pop_size = length(design);
    sample_size = attr(design, "sample_size");

    if (sample_id <= pop_size - sample_size + 1) {
      s = design[sample_id:(sample_id + sample_size - 1)];
      sort(s);
      return(s);
    } else {
      s = design[c(1:(sample_size - pop_size + sample_id - 1), sample_id:pop_size)];
      sort(s);
      return(s);
    }
  } else if (method == "dbd_tc") {
    return(design[, sample_id]);
  } else {
    # Should be unreachable
    stop("method not supported");
  }
}
