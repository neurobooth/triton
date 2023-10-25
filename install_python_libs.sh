#!/usr/bin/env bash
# ================================================================================
# Install Python libraries to be used by the Singularity container for this project.
# ================================================================================

./singularity_exec.sh pipenv install \
numpy \
"scipy>=1.10" \
pandas \
"torch>=2.1" \
torchvision \
torchaudio \
nvidia-pytriton \
lightning \
torchmetrics \
"pydantic>=2.0" \
pyyaml \
tqdm

# These correspond to local code repos; make sure they are bound in env.sh
./singularity_exec.sh pipenv install -e \
/dep/neurobooth-analysis-tools \
/dep/gaze_analysis

# Test the environment
./singularity_exec.sh pipenv run python install_python_libs_test.py
