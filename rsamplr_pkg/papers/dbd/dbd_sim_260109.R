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

##
## Helpers
##
fround = \(x, d = 2L, w = d * 2L + 1L) format(round(x, d), nsmall = d, width = w);
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

##
## Setup population
##
set.seed(251226);
pop_size = 1000L;
sample_size = 50L;
p_cols_max = 20L;
p_cols = 5L;
prob = rep(sample_size / pop_size, pop_size);
xs_base = matrix(runif(pop_size * p_cols_max), ncol = p_cols_max);
xs = xs_base[, 1:p_cols];

##
## Graph iterations: Example 1
## Run the algorithm and store energy at iteration intervals
##
iterations_to = 2000000L;
iterations_by = 50000L;
set.seed(26010901);
iterations_res = dbd_circular_iter(
    sample_size,
    xs,
    annealing_temp    = 0.1,
    annealing_cooling = 0.999,
    max_iter          = iterations_to,
    iter_by           = iterations_by
)
iterations_data = data.frame(
  iterations = seq(iterations_by, iterations_to, iterations_by),
  mean = iterations_res[, 1],
  sd = iterations_res[, 2]
);

ggplot(iterations_data, aes(x = iterations, y = mean, color = "", fill = "")) +
  geom_ribbon(aes(ymin = mean - 2 * sd, ymax = mean + 2 * sd), alpha = 0.2, linewidth = 0.3) +
  geom_line() +
  scale_x_continuous(
    breaks = seq(0, iterations_to, length.out = 11),
    labels = paste0(seq(0, iterations_to, length.out = 11) / 1000, "k")
  ) +
  scale_y_continuous(
    breaks = seq(0.004, 0.014, 0.002),
    limits = c(0.004, 0.014)
  ) +
  gg_defaults
ggsave("papers/dbd/dbs-iterations.pdf", width = 6, height = 4)

# Compare w/ LPM
# Run lpm 10k and store energies
set.seed(26010902);
runs = 10000L;
lpm_samples = matrix(0L, sample_size, runs);
for (r in seq_len(runs)) {
  lpm_samples[, r] = lpm_2(prob, xs);
}
lpm_energy = apply(
  lpm_samples,
  2,
  \(s) spatial_balance_energy_equal(s, xs)
);
lpm_energy_ci = lpm_energy |> (\(.) c(mean(.) - 2.0 * sd(.), mean(.), mean(.) + 2.0 * sd(.)))();

ggplot(data.frame(y = lpm_energy, x = 0), aes(y = y, x = 0, color = "", fill = "")) +
  ## geom_violin(scale = "area", alpha = 0.2) +
  geom_boxplot(alpha = 0.2) +
  scale_x_continuous(breaks = 0, labels = "") +
  scale_y_continuous(
    breaks = seq(0.004, 0.014, 0.002),
    limits = c(0.004, 0.014)
  ) +
  gg_defaults
ggsave("papers/dbd/lpm-energy-boxplot.pdf", width = 3, height = 4)

ggplot(iterations_data, aes(x = iterations, y = mean, color = "", fill = "")) +
  geom_ribbon(aes(ymin = mean - 2 * sd, ymax = mean + 2 * sd), alpha = 0.2, linewidth = 0.6,
  linetype = 2) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = lpm_energy_ci[2], color = cbb_palette[2], linewidth = 1.0, linetype = 1) +
  geom_hline(yintercept = lpm_energy_ci[1], color = cbb_palette[2], linewidth = 0.6, linetype = 2) +
  geom_hline(yintercept = lpm_energy_ci[3], color = cbb_palette[2], linewidth = 0.6, linetype = 2) +
  scale_x_continuous(
    breaks = seq(0, iterations_to, length.out = 11),
    labels = paste0(seq(0, iterations_to, length.out = 11) / 1000, "k")
  ) +
  scale_y_continuous(
    breaks = seq(0.004, 0.009, 0.001),
    limits = c(0.004, 0.009)
  ) +
  gg_defaults

##
## Compare designs using different aux sizes: Example 2
## Compare spatial balance measure(s) of different designs for different aux sizes
##
set.seed(26010903);
p_cols_vec = c(2L, 5L, 10L, 20L);
runs = 10000L;

# Helper fun
design_comp_fn = \(s, prob, data) {
  c(
    spatial_balance_all_equal(s, data)[c(4, 2, 1)],
    sqrt(sum(balance_deviation(s, prob, data)^2))
  )
}

design_comp_res = list(
  srs = tibble(
    p = rep(p_cols_vec, each = runs),
    energy = 0.0,
    lb = 0.0,
    sb = 0.0,
    bal = 0.0
  ),
  lpm = tibble(
    p = rep(p_cols_vec, each = runs),
    energy = 0.0,
    lb = 0.0,
    sb = 0.0,
    bal = 0.0
  ),
  lcube = tibble(
    p = rep(p_cols_vec, each = runs),
    energy = 0.0,
    lb = 0.0,
    sb = 0.0,
    bal = 0.0
  ),
  dbd = tibble(
    p = rep(p_cols_vec, each = pop_size),
    energy = 0.0,
    lb = 0.0,
    sb = 0.0,
    bal = 0.0
  )
);

for (p_k in seq_along(p_cols_vec)) {
  cat("p: ", p_cols_vec[p_k], "\n");
  # Setup aux mats
  xs_p = xs_base[, 1:p_cols_vec[p_k]];
  xb_p = cbind(prob, xs_p);
  offset = runs * (p_k - 1);

  # Run dbd for current size (only once)
  s = dbd_circular(
    sample_size,
    xs_p,
    annealing_temp = 0.1,
    annealing_cooling = 0.999,
    full_design = TRUE,
    max_iter = 1e7
  );

  # Store dbd results
  design_comp_res$dbd[1:pop_size + (p_k - 1) * pop_size, 2:5] = vapply(
    1:pop_size,
    \(i) {
      ss = sequence_to_sample(s, i, sample_size);
      design_comp_fn(ss, prob, xs_p)
    },
    rep(0.0, 4L)
  ) |> t();

  # Run other designs and store
  for (r in seq_len(runs)) {
    s = sample(pop_size, sample_size);
    design_comp_res$srs[r + offset, 2:5] = as.list(design_comp_fn(s, prob, xs_p));

    s = lpm_2(prob, xs_p);
    design_comp_res$lpm[r + offset, 2:5] = as.list(design_comp_fn(s, prob, xs_p));

    s = local_cube(prob, xs_p, xb_p);
    design_comp_res$lcube[r + offset, 2:5] = as.list(design_comp_fn(s, prob, xs_p));
  }
}

# Store data in combined structure
design_comp_combined = bind_rows(design_comp_res, .id = "type") |>
  mutate(p_fac = as.factor(p)) |>
  tidyr::pivot_longer(
      c(energy, lb, sb, bal),
      names_to = "measure",
      values_to = "val"
  )

design_comp_combined |> filter(type != "srs") |>
  ggplot(aes(x = val, color = type, fill = type, linetype = type)) +
  geom_density(alpha = 0.2) +
  facet_wrap(measure ~ p_fac, scales = "free") +
  gg_defaults

design_comp_combined |> filter(measure == "energy", type != "srs") |>
  ggplot(aes(x = val, color = type, fill = type, linetype = type)) +
  geom_density(alpha = 0.2) +
  facet_wrap( ~ p_fac, scales = "free", nrow = 1) +
  scale_x_continuous(n.breaks = 4) +
  gg_defaults +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
ggsave("papers/dbd/simulation-comp-energy.pdf", width = 9, height = 3)

design_comp_combined |> filter(measure == "lb", type != "srs") |>
  ggplot(aes(x = val, color = type, fill = type, linetype = type)) +
  geom_density(alpha = 0.2) +
  facet_wrap( ~ p_fac, scales = "free", nrow = 1) +
  scale_x_continuous(n.breaks = 4) +
  gg_defaults +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
ggsave("papers/dbd/simulation-comp-lb.pdf", width = 9, height = 3)

design_comp_combined |> filter(measure == "sb", type != "srs") |>
  ggplot(aes(x = val, color = type, fill = type, linetype = type)) +
  geom_density(alpha = 0.2) +
  facet_wrap( ~ p_fac, scales = "free", nrow = 1) +
  scale_x_continuous(n.breaks = 4) +
  gg_defaults +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
ggsave("papers/dbd/simulation-comp-sb.pdf", width = 9, height = 3)

design_comp_combined |> filter(measure == "bal", type != "srs") |>
  ggplot(aes(x = val, color = type, fill = type, linetype = type)) +
  geom_density(alpha = 0.2) +
  facet_wrap( ~ p_fac, scales = "free", nrow = 1) +
  scale_x_continuous(n.breaks = 4) +
  gg_defaults +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
ggsave("papers/dbd/simulation-comp-bal.pdf", width = 9, height = 3)

# Print Table 1
design_comp_combined_fn = \(type, p, measure) {
  v = design_comp_res[[type]] |> filter(p == !!p) |> colMeans() |> as.list();
  d = 2;
  if (measure == "energy") {
    d = 4;
  } else if (measure == "lb" || measure == "sb") {
    d = 4;
  }
  fround(v[[measure]], d)
}
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
    design_comp_combined_fn("dbd", p, "bal"), " \\\\\n"
  );
}


##
## Compare designs of different sizes: Example 3
## Compare spatial balance measure(s) of different designs and sample sizees
## Aux size 5
##
p_cols = 5L;
set.seed(26010904);
sample_size_vec = c(100L, 200L);
xs_p = xs_base[, 1:p_cols]
runs = 10000L;

design_comp_res_n = list(
  srs = tibble(
    n = rep(sample_size_vec, each = runs),
    p = p_cols,
    energy = 0.0,
    lb = 0.0,
    sb = 0.0,
    bal = 0.0
  ),
  lpm = tibble(
    n = rep(sample_size_vec, each = runs),
    p = p_cols,
    energy = 0.0,
    lb = 0.0,
    sb = 0.0,
    bal = 0.0
  ),
  lcube = tibble(
    n = rep(sample_size_vec, each = runs),
    p = p_cols,
    energy = 0.0,
    lb = 0.0,
    sb = 0.0,
    bal = 0.0
  ),
  dbd = tibble(
    n = rep(sample_size_vec, each = pop_size),
    p = p_cols,
    energy = 0.0,
    lb = 0.0,
    sb = 0.0,
    bal = 0.0
  )
);

for (n_k in seq_along(sample_size_vec)) {
  nn_k = sample_size_vec[n_k];
  cat("n: ", nn_k, "\n");
  prob = rep(nn_k / pop_size, pop_size);
  xb_p = cbind(prob, xs_p);
  offset = runs * (n_k - 1);

  if (nn_k == sample_size) {
    design_comp_res_n$dbd[1:pop_size + (n_k - 1) * pop_size, 3:6] = design_comp_res$dbd |>
      filter(p == p_cols) |>
      select(energy, lb, sb, bal);
    design_comp_res_n$srs[1:runs + offset, 3:6] = design_comp_res$srs |>
      filter(p == p_cols) |>
      select(energy, lb, sb, bal);
    design_comp_res_n$lpm[1:runs + offset, 3:6] = design_comp_res$lpm |>
      filter(p == p_cols) |>
      select(energy, lb, sb, bal);
    design_comp_res_n$lcube[1:runs + offset, 3:6] = design_comp_res$lcube |>
      filter(p == p_cols) |>
      select(energy, lb, sb, bal);
    next;
  }

  # Run dbd once and store
  s = dbd_circular(
    nn_k,
    xs_p,
    annealing_temp = 0.1,
    annealing_cooling = 0.999,
    full_design = TRUE,
    max_iter = 1e7
  );
  design_comp_res_n$dbd[1:pop_size + (n_k - 1) * pop_size, 3:6] = vapply(
    1:pop_size,
    \(i) {
      ss = sequence_to_sample(s, i, nn_k);
      design_comp_fn(ss, prob, xs_p)
    },
    rep(0.0, 4L)
  ) |> t();

  # Run and store other designs
  for (r in seq_len(runs)) {
    s = sample(pop_size, nn_k);
    design_comp_res_n$srs[r + offset, 3:6] = as.list(design_comp_fn(s, prob, xs_p));

    s = lpm_2(prob, xs_p);
    design_comp_res_n$lpm[r + offset, 3:6] = as.list(design_comp_fn(s, prob, xs_p));

    s = local_cube(prob, xs_p, xb_p);
    design_comp_res_n$lcube[r + offset, 3:6] = as.list(design_comp_fn(s, prob, xs_p));
  }
}

# Print table 2
design_comp_res_n_fn = \(type, n, measure) {
  v = design_comp_res_n[[type]] |> filter(n == !!n) |> colMeans() |> as.list();
  d = 2;
  if (measure == "energy") {
    d = 4;
  } else if (measure == "lb" || measure == "sb") {
    d = 4;
  }
  fround(v[[measure]], d)
}
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
    design_comp_res_n_fn("dbd", p, "bal"), " \\\\\n"
  );
}

##
## MEUSE DATA: Example 3
##
set.seed(26010905);
# Download meuse data from gstat
# Standardize aux variables, keep y-values for eval
meuse_data = {
  temp_file = tempfile();
  meuse_url = "https://github.com/cran/gstat/raw/master/data/meuse.all.rda";
  download.file(meuse_url, temp_file, mode = "wb");
  load(temp_file);
  unlink(temp_file);
  meuse.all
} |> as_tibble() |>
  mutate(
    y_zinc = zinc, y_lead = lead, y_cadmium = cadmium, y_copper = copper, y_elev = elev, y_om = om
  ) |>
  select(
    x, y, elev, om, copper,
    y_zinc, y_lead, y_cadmium, y_copper, y_elev, y_om
  ) |>
  filter(!if_any(everything(), is.na)) |>
  mutate(
    x = as.vector(scale(x)),
    y = as.vector(scale(y)),
    elev = as.vector(scale(elev)),
    om = as.vector(scale(om)),
    copper = as.vector(scale(copper))
  )

meuse_n = 20L;
meuse_p = rep(meuse_n / nrow(meuse_data), nrow(meuse_data));
meuse_xs = meuse_data[, 1:5] |> as.matrix();
meuse_xb = cbind(meuse_p, meuse_xs);
meuse_param = meuse_data[, 6:11] |> colSums() |> as.list()

meuse_res = tibble(
  type = rep("", runs * 3 + nrow(meuse_data)),
  zinc = 0.0,
  c_zinc = 0.0,
  lead = 0.0,
  c_lead = 0.0,
  cadmium = 0.0,
  c_cadmium = 0.0,
  copper = 0.0,
  c_copper = 0.0,
  elev = 0.0,
  c_elev = 0.0,
  om = 0.0,
  c_om = 0.0,
  energy = 0.0,
  sb = 0.0,
  lb = 0.0
)
meuse_hv = \(y, xmat, p, param, srs = FALSE, pop = 1) {
  hat = sum(y) / p[1];

  if (srs) {
    v = sqrt((1.0 - p[1]) / p[1] * var(y) * pop);
  } else {
    v = sqrt(local_mean_variance(y, p, xmat, 2));
  }


  c(hat, (hat - 1.96 * v < param && param < hat + 1.96 * v) * 1.0)
}
meuse_res_fn = \(s) {
  dt = meuse_data[s, ];
  c(
    meuse_hv(dt$y_zinc, dt[, 1:5], meuse_p[s], meuse_param$y_zinc),
    meuse_hv(dt$y_lead, dt[, 1:5], meuse_p[s], meuse_param$y_lead),
    meuse_hv(dt$y_cadmium, dt[, 1:5], meuse_p[s], meuse_param$y_cadmium),
    meuse_hv(dt$y_copper, dt[, 1:5], meuse_p[s], meuse_param$y_copper),
    meuse_hv(dt$y_elev, dt[, 1:5], meuse_p[s], meuse_param$y_elev),
    meuse_hv(dt$y_om, dt[, 1:5], meuse_p[s], meuse_param$y_om),
    spatial_balance_all_equal(s, meuse_xs)[c(4, 2, 1)]
  )
}
meuse_res_fn2 = \(s) {
  dt = meuse_data[s, ];
  pop_n = nrow(meuse_data);
  c(
    meuse_hv(dt$y_zinc, dt[, 1:5], meuse_p[s], meuse_param$y_zinc, TRUE, pop_n),
    meuse_hv(dt$y_lead, dt[, 1:5], meuse_p[s], meuse_param$y_lead, TRUE, pop_n),
    meuse_hv(dt$y_cadmium, dt[, 1:5], meuse_p[s], meuse_param$y_cadmium, TRUE, pop_n),
    meuse_hv(dt$y_copper, dt[, 1:5], meuse_p[s], meuse_param$y_copper, TRUE, pop_n),
    meuse_hv(dt$y_elev, dt[, 1:5], meuse_p[s], meuse_param$y_elev, TRUE, pop_n),
    meuse_hv(dt$y_om, dt[, 1:5], meuse_p[s], meuse_param$y_om, TRUE, pop_n),
    spatial_balance_all_equal(s, meuse_xs)[c(4, 2, 1)]
  )
}

# Run dbd
meuse_dbd = dbd_circular(
  meuse_n,
  meuse_xs,
  annealing_temp = 0.1,
  annealing_cooling = 0.999,
  full_design = TRUE,
  max_iter = 1e7
);
meuse_res$type[seq_len(nrow(meuse_data))] = "dbd";
meuse_res[seq_len(nrow(meuse_data)), -1] = vapply(
  seq_len(nrow(meuse_data)),
  \(x) {
    ss = sequence_to_sample(meuse_dbd, x, meuse_n);
    meuse_res_fn(ss)
  },
  rep(0.0, 15)
) |> t()

# Run other designs
meuse_r = nrow(meuse_data);
while (meuse_r < nrow(meuse_res)) {
  meuse_r = meuse_r + 1;
  s = sample(nrow(meuse_data), meuse_n);
  meuse_res$type[meuse_r] = "srs";
  meuse_res[meuse_r, -1] = as.list(meuse_res_fn2(s));

  meuse_r = meuse_r + 1;
  s = lpm_2(meuse_p, meuse_xs);
  meuse_res$type[meuse_r] = "lpm";
  meuse_res[meuse_r, -1] = as.list(meuse_res_fn(s));

  meuse_r = meuse_r + 1;
  s = local_cube(meuse_p, meuse_xs, meuse_xb);
  meuse_res$type[meuse_r] = "lcube";
  meuse_res[meuse_r, -1] = as.list(meuse_res_fn(s));
}

# Print table 3
meuse_cat_fn = \(type) {
  a = meuse_res |>
    filter(type == !!type) |>
    mutate(
      zi = (zinc - mean(zinc))^2,
      le = (lead - mean(lead))^2,
      ca = (cadmium - mean(cadmium))^2,
      co = (copper - mean(copper))^2,
      el = (elev - mean(elev))^2,
      om = (om - mean(om))^2,
    ) |>
    select(zi, le, ca, co, el, om) |>
    colMeans() |>
    sqrt() / unlist(meuse_param)
  b = meuse_res |>
    filter(type == !!type) |>
    select(energy, sb, lb) |>
    colMeans()
  cat(type, " & ");
  cat(
    fround(c(a, b), 3),
    sep = " & "
  );
  cat("\\\\\n");
}
meuse_cat_fn2 = \(type) {
  a = meuse_res |>
    filter(type == !!type) |>
    select(c_zinc, c_lead, c_cadmium, c_copper, c_elev, c_om) |>
    colMeans()
  cat(type, " & ");
  cat(
    fround(a, 3),
    sep = " & "
  );
  cat("\\\\\n");
}

cat("Method & Zinc & Lead & Cadmium & Copper & Elev & Om & \\mathcal{E} & SB & LB \\\\\n");
meuse_cat_fn("srs")
meuse_cat_fn("lpm")
meuse_cat_fn("lcube")
meuse_cat_fn("dbd")

cat("Method & Zinc & Lead & Cadmium & Copper & Elev & Om \\\\\n");
meuse_cat_fn2("srs")
meuse_cat_fn2("lpm")
meuse_cat_fn2("lcube")
meuse_cat_fn2("dbd")


##
## Save data
##
save.image("papers/dbd/dbd_sim_260109.Rdata");
