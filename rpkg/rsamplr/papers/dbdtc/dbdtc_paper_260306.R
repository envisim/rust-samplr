library("ggplot2");
library("tibble");
library("dplyr");

##
## Compile pkg
##
savvy::savvy_update();
## devtools::document();
pkgbuild::compile_dll(debug = FALSE, force = TRUE);
devtools::document(roclets = NULL);

## Helpers
fround = \(x, d = 2L, w = d * 2L + 1L) format(round(x, d), nsmall = d, width = w, scientific = FALSE);
cbb_palette = c(
  "#000000",
  "#E69F00",
  "#56B4E9",
  "#009E73",
  "#F0E442",
  "#0072B2",
  "#D55E00",
  "#CC79A7"
);
gg_defaults = list(
  scale_color_manual(values = cbb_palette),
  scale_fill_manual(values = cbb_palette),
  guides(fill = "none", color = "none", linetype = "none"),
  labs(x = NULL, y = NULL),
  theme_bw()
);

## Load dbd results
load("papers/dbd/dbd_sim_260109.Rdata");

##
## Graph iterations: Example 2
## Run the algorithm and store energy at iteration intervals
##
sample_size = 50L;
p_cols = 5L;
prob = rep(sample_size / pop_size, pop_size);
xs = xs_base[, 1:p_cols];

iterations_to = 10000000L;
iterations_by = 50000L;
set.seed(260101);
iterations_res_dbd = dbd_circular_iter(
    sample_size,
    xs,
    annealing_temp    = 0.1,
    annealing_cooling = 0.999,
    max_iter          = iterations_to,
    iter_by           = iterations_by
)
iterations_res_bc = dbd_tc_iter(
    sample_size,
    xs,
    annealing_temp    = 0.1,
    annealing_cooling = 0.999,
    max_iter          = iterations_to,
    iter_by           = iterations_by,
    spatial_init      = FALSE
);
iterations_data_bc = bind_rows(
  tibble(
    iterations = seq(iterations_by, iterations_to, iterations_by),
    mean = iterations_res_dbd[, 1],
    sd = iterations_res_dbd[, 2],
    type = "dbd"
  ),
  tibble(
    iterations = seq(iterations_by, iterations_to, iterations_by),
    mean = iterations_res_bc[, 1],
    sd = iterations_res_bc[, 2],
    type = "dbdtc"
  )
);

iterations_data_bc |>
  mutate(ymin = mean - 2 * sd, ymax = mean + 2 * sd) |>
  ggplot(aes(
    x = iterations, y = mean, ymin = ymin, ymax = ymax,
    color = type, fill = type, linetype = type
  )) +
  geom_ribbon(alpha = 0.2, linewidth = 0.3) +
  geom_line() +
  scale_x_continuous(
    breaks = seq(0, iterations_to, length.out = 6),
    labels = paste0(seq(0, iterations_to, length.out = 6) / 1000000, "M")
  ) +
  gg_defaults +
  scale_fill_manual(values = cbb_palette[c(1, 4)]) +
  scale_color_manual(values = cbb_palette[c(1, 4)]) +
  scale_linetype_manual(values = c(1, 4))
ggsave("papers/dbdtc/bc-iterations.pdf", width = 6, height = 3)

##
##
## Variance of warm-start vs not
## Current impl. takes about 1.8 sec
##
set.seed(26010126);
runs = 100L
tc_variance = bind_rows(
  lapply(c(TRUE, FALSE), \(spatial_init) {
    energies = sapply(seq_len(runs), \(r) {
      ds = dbdtc(
        sample_size,
        xs,
        annealing_temp    = 0.1,
        annealing_cooling = 0.999,
        max_iter          = iterations_to,
        spatial_init      = spatial_init,
        full_design       = TRUE
      );
      buckets = attr(ds, "number_of_samples");

      energies = sapply(1:buckets, \(s) {
        spatial_balance_energy_equal(draw(ds, s), xs, sample_size)
      });

      c(mean(energies), sd(energies))
    }) |> t();

    tibble(
      spatial_init = spatial_init,
      mean = energies[, 1],
      sd = energies[, 2]
    )
  })
);

tc_variance |>
  group_by(spatial_init) |>
  summarize(m = mean(mean), rmse = sqrt(var(mean)), rrmse = rmse / m, sdm = mean(sd))

##
## Compare designs using different aux sizes: Example 2
## Compare spatial balance measure(s) of different designs for different aux sizes
##
p_cols_vec = c(2L, 5L, 10L, 20L);
runs = 10000L;

design_comp_res$bc = tibble(
  p = rep(p_cols_vec, each = pop_size / sample_size),
  energy = 0.0,
  lb = 0.0,
  sb = 0.0,
  bal = 0.0
);

# Run dbdtc and combine with results obtained for dbd
set.seed(260102);
for (p_k in seq_along(p_cols_vec)) {
  cat("p: ", p_cols_vec[p_k], "\n");
  xs_p = xs_base[, 1:p_cols_vec[p_k]];

  # Store dbdtc results
  s = dbdtc(
    sample_size,
    xs_p,
    annealing_temp = 0.1,
    annealing_cooling = 0.999,
    full_design = TRUE,
    max_iter = 1e7,
    spatial_init = FALSE
  );
  design_comp_res$bc[1:20 + (p_k - 1) * 20, 2:5] =
    apply(s, 2, design_comp_fn, prob, xs_p) |> t();
}

design_comp_combined = bind_rows(design_comp_res, .id = "type") |>
  mutate(
    p_fac = as.factor(p),
    type = factor(type, levels = c("dbd", "lcube", "lpm", "bc"))
  ) |>
  tidyr::pivot_longer(
      c(energy, lb, sb, bal),
      names_to = "measure",
      values_to = "val"
  ) |>
  mutate(measure = factor(
    measure,
    levels = c("energy", "sb", "lb", "bal"),
    labels = c("Energy","SB", "LB", "BD")
  ))

design_comp_combined |> filter(type != "srs") |>
  ggplot(aes(x = val, color = type, fill = type, linetype = type)) +
  geom_density(alpha = 0.2) +
  scale_x_continuous(n.breaks = 4) +
  ## facet_wrap(measure ~ p_fac, scales = "free", labeller = \(l) label_value(l[2])) +
  facet_wrap(
    measure ~ p_fac,
    scales = "free",
    labeller = \(l) label_value(l, multi_line = FALSE)
  ) +
  gg_defaults +
  ## guides(fill = "legend") +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
ggsave("papers/dbdtc/bc-simulation-comp-all.pdf", width = 9, height = 10)

# Print table 1
cat("\\toprule Dims & Method & $\\mathcal{E}$ &  SB &  LB & MSE \\\\\n");
for (p in p_cols_vec) {
  cat(
    p, " & SRS   & ",
    design_comp_combined_fn("srs", p, "energy"), " & ",
    design_comp_combined_fn("srs", p, "sb"), " & ",
    design_comp_combined_fn("srs", p, "lb"), " & ",
    design_comp_combined_fn("srs", p, "bal"), " \\\\\n",
    "  & LPM   &",
    design_comp_combined_fn("lpm", p, "energy"), " & ",
    design_comp_combined_fn("lpm", p, "sb"), " & ",
    design_comp_combined_fn("lpm", p, "lb"), " & ",
    design_comp_combined_fn("lpm", p, "bal"), " \\\\\n",
    "  & LCUBE &",
    design_comp_combined_fn("lcube", p, "energy"), " & ",
    design_comp_combined_fn("lcube", p, "sb"), " & ",
    design_comp_combined_fn("lcube", p, "lb"), " & ",
    design_comp_combined_fn("lcube", p, "bal"), " \\\\\n",
    "  & DBD   &",
    design_comp_combined_fn("dbd", p, "energy"), " & ",
    design_comp_combined_fn("dbd", p, "sb"), " & ",
    design_comp_combined_fn("dbd", p, "lb"), " & ",
    design_comp_combined_fn("dbd", p, "bal"), " \\\\\n",
    "  & DBD-TC &",
    design_comp_combined_fn("bc", p, "energy"), " & ",
    design_comp_combined_fn("bc", p, "sb"), " & ",
    design_comp_combined_fn("bc", p, "lb"), " & ",
    design_comp_combined_fn("bc", p, "bal"), " \\\\\n"
  );
}

##
## Compare designs of different sizes: Example 3
## Compare spatial balance measure(s) of different designs and sample sizees
## Aux size 5
##
p_cols = 5L;
sample_size_vec = c(100L, 200L);
s_reps = pop_size / sample_size_vec |> as.integer();
xs_p = xs_base[, 1:p_cols]
runs = 10000L;

design_comp_res_n$bc = tibble(
  n = rep(sample_size_vec, times = s_reps),
  p = p_cols,
  energy = 0.0,
  lb = 0.0,
  sb = 0.0,
  bal = 0.0
);

# Run dbdtc and combine with results obtained for dbd
set.seed(260103);
for (n_k in seq_along(sample_size_vec)) {
  nn_k = sample_size_vec[n_k];
  cat("n: ", nn_k, "\n");
  prob = rep(nn_k / pop_size, pop_size);

  # Run dbd once and store
  s = dbdtc(
    nn_k,
    xs_p,
    annealing_temp = 0.1,
    annealing_cooling = 0.999,
    full_design = TRUE,
    max_iter = 1e7,
    spatial_init = FALSE
  );
  design_comp_res_n$bc[1:s_reps[n_k] + ifelse(n_k == 1, 0, sum(s_reps[1:(n_k - 1)])), 3:6] =
    apply(s, 2, design_comp_fn, prob, xs_p) |> t();
}

# Print table 2
cat("\\toprule Size & Method & $\\mathcal{E}$ &  SB &  LB & MSE \\\\\n");
for (p in sample_size_vec) {
  cat(
    p, " & SRS   & ",
    design_comp_res_n_fn("srs", p, "energy"), " & ",
    design_comp_res_n_fn("srs", p, "sb"), " & ",
    design_comp_res_n_fn("srs", p, "lb"), " & ",
    design_comp_res_n_fn("srs", p, "bal"), " \\\\\n",
    "  & LPM   &",
    design_comp_res_n_fn("lpm", p, "energy"), " & ",
    design_comp_res_n_fn("lpm", p, "sb"), " & ",
    design_comp_res_n_fn("lpm", p, "lb"), " & ",
    design_comp_res_n_fn("lpm", p, "bal"), " \\\\\n",
    "  & LCUBE &",
    design_comp_res_n_fn("lcube", p, "energy"), " & ",
    design_comp_res_n_fn("lcube", p, "sb"), " & ",
    design_comp_res_n_fn("lcube", p, "lb"), " & ",
    design_comp_res_n_fn("lcube", p, "bal"), " \\\\\n",
    "  & DBD   &",
    design_comp_res_n_fn("dbd", p, "energy"), " & ",
    design_comp_res_n_fn("dbd", p, "sb"), " & ",
    design_comp_res_n_fn("dbd", p, "lb"), " & ",
    design_comp_res_n_fn("dbd", p, "bal"), " \\\\\n",
    "  & DBD-TC &",
    design_comp_res_n_fn("bc", p, "energy"), " & ",
    design_comp_res_n_fn("bc", p, "sb"), " & ",
    design_comp_res_n_fn("bc", p, "lb"), " & ",
    design_comp_res_n_fn("bc", p, "bal"), " \\\\\n"
  );
}

##
## MEUSE DATA: Example 4
##
# Run dbdtc and combine with results from dbd
set.seed(260104);
meuse_bc = dbdtc(
  meuse_n,
  meuse_xs,
  annealing_temp = 0.1,
  annealing_cooling = 0.999,
  full_design = TRUE,
  max_iter = 1e7,
  spatial_init = FALSE
);
meuse_max = nrow(meuse_res);
meuse_bc_res = apply(meuse_bc, 2, meuse_res_fn) |> t() |>
  as_tibble() |> mutate(type = "bc", .before = 1);
colnames(meuse_bc_res) = colnames(meuse_res);
meuse_res = meuse_res |> bind_rows(meuse_res, meuse_bc_res);

# Print table 3
cat("Method & Zinc & Lead & Cadmium & Copper & Elev & Om & \\mathcal{E} & SB & LB \\\\\n");
meuse_cat_fn("srs")
meuse_cat_fn("lpm")
meuse_cat_fn("lcube")
meuse_cat_fn("dbd")
meuse_cat_fn("bc")

cat("Method & Zinc & Lead & Cadmium & Copper & Elev & Om \\\\\n");
meuse_cat_fn2("srs")
meuse_cat_fn2("lpm")
meuse_cat_fn2("lcube")
meuse_cat_fn2("dbd")
meuse_cat_fn2("bc")
