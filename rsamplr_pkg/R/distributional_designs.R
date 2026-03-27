#' Distributionally balanced designs
#'
#' @name Distributionally balanced designs
#' @rdname distributional_designs
#' @description
#' Construct distributionally balanced sampling designs.
#'
#' @param sample_size The size of the desired sample.
#' @param spread_mat A matrix of spreading covariates.
#' @inheritDotParams .sampling_defaults -bucket_size
#' @inheritDotParams .dbd_defaults
#'
#' @returns A `dbd` object containing all possible samples in the design
#'
#' @references
#' Grafström, A., & Prentius, W. (2026).
#' Distributionally balanced sampling designs.
#' arXiv preprint arXiv:2603.11916.
#'
#' Grafström, A., & Prentius, W. (2026).
#' Distributionally balanced sampling designs via minimum tactical configurations.
#' arXiv preprint arXiv:2603.24439.
#'
#' @examples
#' set.seed(12345);
#' N = 1000L;
#' n = 100L;
#' prob = rep(n / N, N);
#' xs = matrix(runif(N * 2), ncol = 2);
#'
#' # Construct circular dbd design
#' design = dbd_circular(
#'   n,
#'   xs,
#'   annealing_temp = 0.1,
#'   annealing_cooling = 0.999,
#'   max_iter = 1e6L
#' );
#' s = draw(design); # Draw sample from design
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
#' # Construct dbd design using tactical configuration
#' design = dbd_tc(
#'   n,
#'   xs,
#'   annealing_temp = 0.1,
#'   annealing_cooling = 0.999,
#'   max_iter = 1e6L
#' );
#' s = draw(design); # Draw sample from design
#' plot(xs[, 1], xs[, 2], pch = ifelse(sample_to_indicator(s, N), 19, 1));
#'
NULL

#' Distributionally balanced designs defaults
#' Calls [.sampling_defaults()]
#'
#' @param annealing_temp The initial temperature to use in simulated annealing
#' @param annealing_cooling The annealing_cooling rate tu use in simulated annealing
#' @param spatial_init If `TRUE` a spatial initialization strategy is used
#'
.dbd_defaults = function(
  annealing_temp = 0.1,
  annealing_cooling = 0.999,
  spatial_init = FALSE,
  ...
) {
  args = modifyList(
    .sampling_defaults(...),
    list(
      annealing_temp    = as.double(annealing_temp),
      annealing_cooling = as.double(annealing_cooling),
      spatial_init      = as.logical(spatial_init)
    )
  );

  if (!(0.0 < args$annealing_temp)) {
    warning("annealing_temp is non-positive ... using default");
    args$annealing_temp = 0.1;
  }

  if (!(0.0 < args$annealing_cooling && args$annealing_cooling < 1.0)) {
    warning("annealing_cooling is not in (0.0, 1.0) ... using default");
    args$annealing_cooling = 0.999;
  }

  args
}

#' Distributionally balanced designs wrapper
#' Calls [.dbd_defaults()]
#'
#' @param method The method name as defined by the rust wrapper
#' @param sample_size The integer-valued desired sample size
#' @param spread_mat A matrix of auxiliary variables
#'
#' @noRd
.distributionally_balanced_wrapper = function(
  method,
  sample_size,
  spread_mat,
  ...
) {
  args = .dbd_defaults(...);

  spread_mat = as.matrix(spread_mat);
  sample_size = as.integer(sample_size);

  if (!(0L < sample_size && sample_size < nrow(spread_mat))) {
    stop("sample_size must be positive and less than the population size");
  }

  s = rust_distributionally_balanced_design(
    sample_size,
    spread_mat,
    args$annealing_temp,
    args$annealing_cooling,
    args$max_iter,
    args$spatial_init,
    method
  );

  if (method == "dbd") {
    attr(s, "number_of_samples") = length(s);
  } else if (method == "dbdtc") {
    s = matrix(s, sample_size);
    attr(s, "number_of_samples") = ncol(s);
  } else {
    # Should be unreachable
    stop("method not supported");
  }

  class(s) = c("dbd", class(s));
  attr(s, "method") = method;
  attr(s, "sample_size") = sample_size;

  for (arg in names(args)) {
    attr(s, arg) = args[[arg]];
  }

  s
}

#' @describeIn distributional_designs Distributionally Balanced Sampling as circular sequence
#' @export
dbd_circular = function(
  sample_size,
  spread_mat,
  ...
) {
  .distributionally_balanced_wrapper("dbd_circular", sample_size, spread_mat, ...)
}

#' @describeIn distributional_designs Distributionally Balanced Sampling using Tactical Configuration
#' @export
dbd_tc = function(
  sample_size,
  spread_mat,
  ...
) {
  .distributionally_balanced_wrapper("dbd_tc", sample_size, spread_mat, ...)
}

#' Wrapper
#' @noRd
.distributionally_balanced_iter_wrapper = function(
  method,
  sample_size,
  spread_mat,
  iter_by = 100L,
  ...
) {
  args = .dbd_defaults(...);

  iter_by = as.integer(iter_by);

  if (!(length(iter_by) == 1 && 0 < iter_by && iter_by <= args$max_iter)) {
    warnings("iter_by is not within within (0, max_iter) ... using default");
    iter_by = min(100L, args$max_iter);
  }


  s = rust_distributionally_balanced_design_iter(
    sample_size,
    as.matrix(spread_mat),
    args$annealing_temp,
    args$annealing_cooling,
    args$spatial_init,
    args$max_iter,
    iter_by,
    method
  );

  matrix(s, ncol = 2, byrow = TRUE)
}

#' Iterative evaluation of distributional designs
#'
#' @name Iterative evaluation of distributional designs
#' @rdname distributional_designs_iter
#' @description
#' Evaluates the energy distance along a simulated annealing scheme.
#'
#' @param sample_size The size of the desired sample.
#' @param spread_mat A matrix of spreading covariates.
#' @inheritDotParams .sampling_defaults
#' @inheritDotParams .dbd_defaults
#' @param max_iter The number of iterations to run
#' @param iter_by The reporting interval
#'
#' @returns A matrix with the average energies and the standard deviations of the energies as
#' columns, per the reporting intervals.
#' @export
dbd_circular_iter = function(
  sample_size,
  spread_mat,
  ...
) {
  .distributionally_balanced_iter_wrapper("dbd_circular", sample_size, spread_mat, ...)
}

#' @describeIn distributional_designs_iter Iterative evolution of distributional design using
#' Tactical Configuration
#' @export
dbd_tc_iter = function(
  sample_size,
  spread_mat,
  ...
) {
  .distributionally_balanced_iter_wrapper("dbd_tc", sample_size, spread_mat, ...)
}

