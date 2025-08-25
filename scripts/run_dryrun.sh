#!/usr/bin/env bash
set -euo pipefail

# Use your venv Python if present; otherwise fall back to python3
PY="${PY:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

echo "Using Python: $PY"

# 1) Generate synthetic S1D FITS
"$PY" scripts/synth_night.py

# 2) Run the pipeline on the synthetic night
# If Python can’t resolve the package, use the fallback command below.
"$PY" -m mvt.run_mvt_espre --cfg configs/dryrun_naid.yaml
# Fallback (uncomment if needed):
# PYTHONPATH=. "$PY" mvt/run_mvt_espre.py --cfg configs/dryrun_naid.yaml
