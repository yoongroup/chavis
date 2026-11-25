#!/usr/bin/env bash
set -euo pipefail
PKG="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
cd -- "$PKG"
exec "${CHAVIS_PYTHON:-python3}" -B -m ultra_fast_v1.launch "$@"
