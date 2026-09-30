mod rpkg

[default]
_default:
  @just --list --list-submodules

# Run check
rust-check: && rpkg::rust-check
  cargo +1.85.1 check

# Run clippy
clippy: && rpkg::clippy
  cargo clippy

# Run rust test
test: clippy && rpkg::test
  cargo test

# Run rust tests (doc)
test-doc: clippy && rpkg::test-doc
  cargo test --doc

# Run rust tests (examples)
test-examples: clippy && rpkg::test-examples
  cargo test --examples
