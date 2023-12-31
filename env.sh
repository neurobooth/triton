#!/usr/bin/env bash
# ================================================================================
# Define common project variables for use in other scripts
# ================================================================================

export PROJECT_DIR="/space/drwho/3/neurobooth/applications/triton_server/"
export PYTHON_IMAGE="/vast/neurobooth/sif/python_3.10.sif"

# Build up a string of directories that should be exposed to the singularity VM via --bind
BIND="/space/neo/3/neurobooth/applications/neurobooth-analysis-tools:/dep/neurobooth-analysis-tools"
BIND="$BIND,/space/neo/4/analyses/bro7_gaze:/dep/gaze_analysis"
BIND="$BIND,$PROJECT_DIR/.workspace:/workspace"
export BIND
