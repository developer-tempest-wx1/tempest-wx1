#!/usr/bin/env bash
# WX-1 WSL2 Environment Configuration
# Sources Conda Tempest, sets ext4 target and scratch directories, sets LD_LIBRARY_PATH for PyO3

export CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3/envs/Tempest}"
export PATH="$HOME/.cargo/bin:$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Test and scratch locations: set WX1_SCRATCH to a high-capacity data partition
export WX1_TESTS_DIR="${WX1_TESTS_DIR:-$HOME/WX-1-TESTS}"
export WX1_SCRATCH="${WX1_SCRATCH:-$WX1_TESTS_DIR/scratch}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WX1_TESTS_DIR/target}"

mkdir -p "$WX1_TESTS_DIR" "$WX1_SCRATCH" "$CARGO_TARGET_DIR" "$WX1_TESTS_DIR/reports/logs"
