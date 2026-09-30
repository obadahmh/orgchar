#!/bin/bash
# Runs orgchar with the FinalSpark SDK environment (ncp), from anywhere.
PY="${ORGCHAR_PYTHON:-$HOME/miniconda3/envs/ncp/bin/python}"
cd "$(dirname "$(readlink -f "$0")")" && exec "$PY" -m orgchar "$@"
