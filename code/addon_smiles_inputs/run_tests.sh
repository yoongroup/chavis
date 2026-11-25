#!/usr/bin/env bash
# Run the add-on tests. CPU only; writes only inside this folder (results/). Needs Python 3 with RDKit.
# Choose the interpreter with CHAVIS_PYTHON (default: python3).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
PY="${CHAVIS_PYTHON:-python3}"
cd "$(dirname "$0")"
"$PY" -B -m unittest discover -s tests -v
