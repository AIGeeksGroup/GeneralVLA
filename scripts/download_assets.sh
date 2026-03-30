#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRETRAIN_DIR="$ROOT_DIR/vendor/GeneralVLA/pretrain_model"
SEGAGENT_DIR="$PRETRAIN_DIR/segagent/zzzmmz/SegAgent-Model"
HF_REPO_URL="https://huggingface.co/AIGeeksGroup/GeneralVLA"
LOCAL_PRETRAIN="${GENERALVLA_PRETRAIN_SOURCE:-}"
USED_LOCAL_PRETRAIN=0

mkdir -p "$(dirname "$SEGAGENT_DIR")"

copy_tree_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -e "$dst" ]]; then
    echo "Skip existing: $dst"
    return 0
  fi
  if [[ -e "$src" ]]; then
    echo "Copying local asset tree: $src -> $dst"
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
    return 0
  fi
  return 1
}

if [[ -n "$LOCAL_PRETRAIN" ]]; then
  echo "Using local asset source: $LOCAL_PRETRAIN"
  copy_tree_if_missing "$LOCAL_PRETRAIN/LISA-7B-v1-explanatory" "$PRETRAIN_DIR/LISA-7B-v1-explanatory"
  copy_tree_if_missing "$LOCAL_PRETRAIN/clip-vit-large-patch14" "$PRETRAIN_DIR/clip-vit-large-patch14"
  copy_tree_if_missing "$LOCAL_PRETRAIN/segagent" "$PRETRAIN_DIR/segagent"
  copy_tree_if_missing "$LOCAL_PRETRAIN/checkpoints" "$PRETRAIN_DIR/checkpoints"
  if [[ -f "$LOCAL_PRETRAIN/sam_vit_h_4b8939.pth" && ! -f "$PRETRAIN_DIR/sam_vit_h_4b8939.pth" ]]; then
    echo "Copying local asset file: $LOCAL_PRETRAIN/sam_vit_h_4b8939.pth -> $PRETRAIN_DIR/sam_vit_h_4b8939.pth"
    mkdir -p "$PRETRAIN_DIR"
    cp -a "$LOCAL_PRETRAIN/sam_vit_h_4b8939.pth" "$PRETRAIN_DIR/sam_vit_h_4b8939.pth"
  fi
  USED_LOCAL_PRETRAIN=1
fi

if [[ "$USED_LOCAL_PRETRAIN" -eq 1 ]]; then
  echo "Skip Hugging Face clone because local assets were provided."
elif [[ ! -d "$PRETRAIN_DIR/.git" ]]; then
  if [[ -e "$PRETRAIN_DIR" && -n "$(find "$PRETRAIN_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    echo "Expected a cloned Hugging Face repo at $PRETRAIN_DIR, but found a non-git directory."
    echo "Remove or rename it, then rerun this script."
    exit 1
  fi
  rm -rf "$PRETRAIN_DIR"
  echo "Cloning model assets from Hugging Face..."
  git clone "$HF_REPO_URL" "$PRETRAIN_DIR"
else
  echo "Skip existing Hugging Face asset repo: $PRETRAIN_DIR"
fi

if [[ ! -d "$SEGAGENT_DIR" ]]; then
  mkdir -p "$SEGAGENT_DIR"
fi

if [[ -f "$ROOT_DIR/SimSun.ttf" && ! -f "$SEGAGENT_DIR/SimSun.ttf" ]]; then
  cp -a "$ROOT_DIR/SimSun.ttf" "$SEGAGENT_DIR/SimSun.ttf"
fi

echo "Asset localization complete."
echo "Run:"
echo "  PYTHONPATH=src python -m robot_memory_vla.app.main --preflight --config-dir ./configs"
