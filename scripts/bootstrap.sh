#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "robotvla39"; then
      conda activate robotvla39
    fi
  fi
fi

python -m pip install -e '.[dev]'

echo "Bootstrap complete."
echo "Next steps:"
echo "  1. bash scripts/download_assets.sh"
echo "  2. PYTHONPATH=src pytest -q"
echo "  3. PYTHONPATH=src python -m robot_memory_vla.app.main --preflight --config-dir ./configs"
