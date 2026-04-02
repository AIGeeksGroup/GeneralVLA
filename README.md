# GeneralVLA: Generalizable Vision-Language-Action Models with Knowledge-Guided Trajectory Planning

This is the official repository for the paper:
> **GeneralVLA: Generalizable Vision-Language-Action Models with Knowledge-Guided Trajectory Planning**
>
> Guoqing Ma\*, Siheng Wang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*†, Shan Yu, and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. †Project lead. <sup>#</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/abs/2602.04315) | [Website](https://aigeeksgroup.github.io/GeneralVLA) | [Models](https://huggingface.co/AIGeeksGroup/GeneralVLA) | [HF Paper](https://huggingface.co/papers/2602.04315)

## 1. Repository Layout

Key directories:

- `src/robot_memory_vla/`
  - main project source code
- `configs/`
  - project configuration
- `scripts/`
  - setup and asset download scripts
- `external/graspnet/`
  - repository-local `graspnet` code and checkpoint location

## 2. Installation

```bash
cd <repo-root>
bash scripts/bootstrap.sh
```

If you prefer manual setup:

```bash
cd <repo-root>
conda activate robotvla39  # if you use conda
python -m pip install -e '.[dev]'
```

## 3. Download Assets

GitHub stores code only. Pretrained models and checkpoints are hosted on Hugging Face:

- `https://huggingface.co/AIGeeksGroup/GeneralVLA`

To fetch repository-local model assets:

```bash
cd <repo-root>
git clone https://huggingface.co/AIGeeksGroup/GeneralVLA vendor/GeneralVLA/pretrain_model
```

If you already have a local copy of the assets, place them under:

- `vendor/GeneralVLA/pretrain_model/LISA-7B-v1-explanatory`
- `vendor/GeneralVLA/pretrain_model/segagent/zzzmmz/SegAgent-Model`
- `vendor/GeneralVLA/pretrain_model/sam_vit_h_4b8939.pth`
- `vendor/GeneralVLA/pretrain_model/clip-vit-large-patch14`
- `vendor/GeneralVLA/pretrain_model/checkpoints/v1/checkpoint-rs.tar`

Notes:

- these files are large and require enough disk space
- `scripts/download_assets.sh` clones the same Hugging Face repository into `vendor/GeneralVLA/pretrain_model`
- `configs/models.yaml` already points to these repository-local paths

## 4. Configuration

Configuration files are located in:

- `configs/models.yaml`
- `configs/robot.yaml`
- `configs/runtime.yaml`

The shipped defaults now use repository-relative paths.

### `configs/models.yaml`

Controls:

- vendored source directories
- GeneralVLA model locations
- GraspNet paths
- inference parameters

### `configs/robot.yaml`

Controls:

- real robot TCP endpoint
- gripper defaults
- robot initial pose

### `configs/runtime.yaml`

Controls:

- runtime output directories
- memory file path
- retrieval `top_k`
- whether execution requires operator confirmation

## 5. Verification

### 5.1 Run Tests

```bash
cd <repo-root>
conda activate robotvla39  # if you use conda
pytest -q
```

Current verified result:

```text
33 passed
```

### 5.2 Preflight Check

```bash
cd <repo-root>
conda activate robotvla39  # if you use conda
python -m robot_memory_vla.app.main --preflight --config-dir ./configs
```

Behavior:

- if the repository-local assets are present, `preflight` should pass
- if the assets are missing, `preflight` will report the missing repo-local model paths

## 6. Run the Main Flow

```bash
cd <repo-root>
conda activate robotvla39  # if you use conda
python -m robot_memory_vla.app.main \
  --config-dir ./configs \
  --task "Pick up the bottle cap on the desk and place it on the pink box in the lower-right corner"
```
