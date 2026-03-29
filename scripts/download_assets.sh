#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRETRAIN_DIR="$ROOT_DIR/vendor/GeneralVLA/pretrain_model"
SEGAGENT_DIR="$PRETRAIN_DIR/segagent/zzzmmz/SegAgent-Model"
LOCAL_PRETRAIN="${GENERALVLA_PRETRAIN_SOURCE:-}"

mkdir -p "$PRETRAIN_DIR"
mkdir -p "$(dirname "$SEGAGENT_DIR")"

copy_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -e "$dst" ]]; then
    echo "Skip existing: $dst"
    return 0
  fi
  if [[ -e "$src" ]]; then
    echo "Copying local asset: $src -> $dst"
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
    return 0
  fi
  return 1
}

download_with_python() {
  local repo_id="$1"
  local dst="$2"
  python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
PY
  python - "$repo_id" "$dst" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
dst = sys.argv[2]
snapshot_download(repo_id=repo_id, local_dir=dst, local_dir_use_symlinks=False)
PY
}

copy_if_missing "${LOCAL_PRETRAIN:+$LOCAL_PRETRAIN/}LISA-7B-v1-explanatory" "$PRETRAIN_DIR/LISA-7B-v1-explanatory" || {
  echo "Downloading LISA-7B-v1-explanatory from Hugging Face..."
  download_with_python "xinlai/LISA-7B-v1-explanatory" "$PRETRAIN_DIR/LISA-7B-v1-explanatory"
}

copy_if_missing "${LOCAL_PRETRAIN:+$LOCAL_PRETRAIN/}clip-vit-large-patch14" "$PRETRAIN_DIR/clip-vit-large-patch14" || {
  echo "Downloading clip-vit-large-patch14 from Hugging Face..."
  download_with_python "openai/clip-vit-large-patch14" "$PRETRAIN_DIR/clip-vit-large-patch14"
}

copy_if_missing "${LOCAL_PRETRAIN:+$LOCAL_PRETRAIN/}segagent/zzzmmz/SegAgent-Model" "$SEGAGENT_DIR" || {
  echo "Downloading SegAgent-Model from ModelScope..."
  git clone https://www.modelscope.cn/zzzmmz/SegAgent-Model.git "$SEGAGENT_DIR"
}

copy_if_missing "${LOCAL_PRETRAIN:+$LOCAL_PRETRAIN/}sam_vit_h_4b8939.pth" "$PRETRAIN_DIR/sam_vit_h_4b8939.pth" || {
  echo "Downloading SAM checkpoint..."
  wget -O "$PRETRAIN_DIR/sam_vit_h_4b8939.pth" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
}

if [[ ! -d "$SEGAGENT_DIR" ]]; then
  mkdir -p "$SEGAGENT_DIR"
fi

if [[ -f "$ROOT_DIR/SimSun.ttf" && ! -f "$SEGAGENT_DIR/SimSun.ttf" ]]; then
  cp -a "$ROOT_DIR/SimSun.ttf" "$SEGAGENT_DIR/SimSun.ttf"
fi

echo "Asset localization complete."
echo "Run:"
echo "  PYTHONPATH=src python -m robot_memory_vla.app.main --preflight --config-dir ./configs"
