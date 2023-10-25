#!/usr/bin/env bash
# ================================================================================
# Execute a command in the singularity container
# ================================================================================

source ./env.sh
singularity exec --nv -H "$PROJECT_DIR" --bind "$BIND" "$PYTHON_IMAGE" "$@"
