pkg := 'rsamplr_pkg'
# r_source := pkg + '/R/*'
rust_source := pkg + '/src/rust'
manifest := rust_source + '/Cargo.toml'
vendor := rust_source + '/vendor'
build := 'build'


[default]
_default:
  @just --list rpkg

# Run clippy
clippy:
  cargo clippy --manifest-path {{manifest}}

# Run rust test
test: clippy
  cargo test --manifest-path {{manifest}}

# Run rust tests (doc)
test-doc: clippy
  cargo test --doc

# Run rust tests (examples)
test-examples: clippy
  cargo test --examples

# Clean the environment
clean-all:
	rm -rf {{vendor}} || true

# Build the package source
build: vendor document
  cd {{pkg}} && \
  mkdir -p {{build}} && \
  R -e 'devtools::build(path = "{{build}}"); devtools::build_manual("{{build}}", "{{build}}");'

# Build savvy wrappers
build-wrappers:
  cd {{pkg}} && \
  R -e 'savvy::savvy_update()'

# Construct vendoring
vendor: clean-all
  cd {{pkg}} && \
  Rscript dev/vendoring.R \
  Rscript dev/vendor-authors.R -m {{manifest}} -o ./inst --verbose \
  Rscript dev/vendor-license-note.R

# Generate documentation
document: build-wrappers vendor
  cd {{pkg}} && \
  R -e 'devtools::document()'

# Check a build
check: build
  cd {{pkg}} && \
  R -e 'devtools::check_built("{{build}}", cran = TRUE)'

# Performs sanity check on code
check-sanity:
  cd {{pkg}} && \
  R -e 'devtools::spell_check("{{build}}")'
