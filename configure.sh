#!/usr/bin/env bash

set -e
set -o pipefail

if [ -z "$PYTHON_BIN_PATH" ]; then
  PYTHON_BIN_PATH=$(which python3 || which python || true)
fi

# Set all env variables
CONFIGURE_DIR=$(dirname "$0")
"$PYTHON_BIN_PATH" "${CONFIGURE_DIR}/configure.py" "$@"

echo "Configuration finished"