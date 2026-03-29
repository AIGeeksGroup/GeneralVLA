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

To download or localize model assets:

```bash
cd <repo-root>
export GENERALVLA_PRETRAIN_SOURCE=/path/to/existing/pretrain_model  # optional
bash scripts/download_assets.sh
```

The script populates repository-local paths such as:

- `vendor/GeneralVLA/pretrain_model/LISA-7B-v1-explanatory`
- `vendor/GeneralVLA/pretrain_model/segagent/zzzmmz/SegAgent-Model`
- `vendor/GeneralVLA/pretrain_model/sam_vit_h_4b8939.pth`
- `vendor/GeneralVLA/pretrain_model/clip-vit-large-patch14`

Notes:

- these files are large and require enough disk space
- if disk space is insufficient, the script will fail and `preflight` will continue reporting missing assets

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

