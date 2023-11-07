#!/usr/bin/env bash
# ================================================================================
# Install Python libraries to be used by the Singularity container for this project.
# Requires the presence of a Pipfile specifying the packages to install.
# ================================================================================

# Abort if any command fails
set -e

# Install from the Pipfile
# Make sure local code bases are mapped in the in the --bind argument (see env.sh)
./singularity_exec.sh pipenv install

# Test the environment
./singularity_exec.sh pipenv run python install_python_libs_test.py
