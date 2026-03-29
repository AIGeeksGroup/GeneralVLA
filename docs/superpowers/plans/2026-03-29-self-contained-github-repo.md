# Self-Contained GitHub Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert this integrated project into a GitHub-friendly self-contained repository that no longer depends on `/home/why/reasoning-bank`, `/home/why/GeneralVLA`, or `/home/why/zeroshotpick-main` code directories at runtime.

**Architecture:** Vendor the minimal required source code from the three upstream projects into this repository under repo-owned directories, then switch configs and adapters from absolute external paths to repo-relative defaults resolved from the current repository root. Keep model assets out of git and provide bootstrap/download scripts plus clear docs so clone -> install -> preflight/test/demo works without local private directory layout.

**Tech Stack:** Python 3.9+, setuptools editable install, pytest, YAML config, local vendored Python modules, shell bootstrap scripts.

---

### Task 1: Freeze the current dependency graph

**Files:**
- Modify: `README.md`
- Create: `docs/dependency-audit.md`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add a test that loads shipped config files and asserts they do not contain `/home/why/reasoning-bank`, `/home/why/GeneralVLA`, `/home/why/zeroshotpick-main`, or `/home/why/robot-memory-vla`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_shipped_config_does_not_use_machine_specific_paths -q`
Expected: FAIL because current config still contains absolute machine-specific paths.

- [ ] **Step 3: Document current dependency inventory**

Create `docs/dependency-audit.md` listing:
- runtime code dependencies on external source trees
- runtime model/weight dependencies
- paths in docs/config/tests that still point to the old machine layout

- [ ] **Step 4: Update config-loading code expectations**

Record the target rule in `README.md`: code must run from repo-relative defaults; large model assets may still be downloaded after clone.

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/test_config.py -q`
Expected: existing config tests still pass except the new failing test.

### Task 2: Introduce repo-relative path resolution

**Files:**
- Modify: `src/robot_memory_vla/app/config.py`
- Modify: `src/robot_memory_vla/app/main.py`
- Modify: `src/robot_memory_vla/runtime/models.py`
- Modify: `configs/models.yaml`
- Modify: `configs/runtime.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- repo-relative shipped config values
- config resolution from repository root instead of `/home/why/...`
- runtime data paths living under this repo

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_config.py -q`
Expected: FAIL on repo-relative expectations.

- [ ] **Step 3: Implement minimal repo-root helpers**

Update config code to:
- resolve a repository root from the current package location
- support relative config values by anchoring them to the repository root
- keep explicit absolute paths working when users override them

- [ ] **Step 4: Replace shipped config defaults**

Change shipped YAML files so they point to repo-owned relative paths such as:
- `vendor/reasoning-bank`
- `vendor/GeneralVLA`
- `vendor/zeroshotpick-main`
- `data/...`
- `external/graspnet/...`

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_config.py -q`
Expected: PASS.

### Task 3: Vendor the required source code into the repository

**Files:**
- Create: `vendor/reasoning-bank/...`
- Create: `vendor/GeneralVLA/...`
- Create: `vendor/zeroshotpick-main/...`
- Modify: `src/robot_memory_vla/adapters/reasoning_bank_adapter.py`
- Modify: `src/robot_memory_vla/adapters/generalvla_adapter.py`
- Modify: `src/robot_memory_vla/adapters/zeroshotpick_adapter.py`
- Test: `tests/test_generalvla_adapter.py`
- Test: `tests/test_zeroshotpick_adapter.py`
- Test: `tests/test_memory_store.py`

- [ ] **Step 1: Identify the minimal vendored file set**

List exactly which upstream files are required for:
- reasoning-bank memory integration
- GeneralVLA demo/model imports used by the adapter
- zeroshotpick upper client / apps / pipeline path used by the adapter

- [ ] **Step 2: Copy the minimal file set into repo-owned vendor directories**

Preserve upstream internal relative imports where practical; avoid copying unused notebooks, caches, checkpoints, and generated outputs.

- [ ] **Step 3: Point adapters at vendored defaults**

Update adapters so default module loading uses repo-owned vendor directories instead of external absolute directories.

- [ ] **Step 4: Write or update adapter tests**

Ensure tests verify:
- adapters can import vendored modules via repo-owned defaults
- no adapter requires `/home/why/...` to work

- [ ] **Step 5: Run adapter tests**

Run: `pytest tests/test_memory_store.py tests/test_generalvla_adapter.py tests/test_zeroshotpick_adapter.py -q`
Expected: PASS.

### Task 4: Add bootstrap and asset download workflow

**Files:**
- Create: `scripts/bootstrap.sh`
- Create: `scripts/download_assets.sh`
- Create: `.gitignore`
- Modify: `src/robot_memory_vla/app/config.py`
- Modify: `README.md`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add a config validation test that expects missing large assets to be reported with repo-local target paths, not machine-specific ones.

- [ ] **Step 2: Run the test to verify failure**

Run: `pytest tests/test_config.py::test_validate_app_config_reports_repo_local_missing_assets -q`
Expected: FAIL because messages still reference old paths or missing repo-local conventions.

- [ ] **Step 3: Implement bootstrap scripts**

Create:
- `scripts/bootstrap.sh` for editable install and lightweight setup
- `scripts/download_assets.sh` for downloading or guiding users to required large files into repo-local directories

The script should:
- create expected asset directories
- print clear next actions for gated/manual downloads
- be safe to re-run

- [ ] **Step 4: Update validation messages and docs**

Ensure config validation points users toward repo-local asset locations and the new bootstrap/download scripts.

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/test_config.py -q`
Expected: PASS.

### Task 5: Add a no-hardware local demo path

**Files:**
- Create: `scripts/run_local_demo.sh`
- Create or Modify: `data/demo/...`
- Modify: `src/robot_memory_vla/app/main.py`
- Modify: `README.md`
- Modify: `explain.md`
- Test: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write the failing test**

Add a CLI smoke test for a documented local-demo entry option or script path that does not require robot hardware.

- [ ] **Step 2: Run the test to verify failure**

Run: `pytest tests/test_cli_smoke.py -q`
Expected: FAIL because no local-demo entry path exists yet.

- [ ] **Step 3: Implement minimal local-demo workflow**

Add a documented local demo route that:
- does not require the real TCP robot endpoint
- exercises at least the repository packaging and non-hardware control flow
- can be invoked after clone with repo-local assets

- [ ] **Step 4: Update docs**

Document:
- bootstrap
- tests
- preflight
- local demo
- real robot mode

- [ ] **Step 5: Run smoke tests**

Run: `pytest tests/test_cli_smoke.py -q`
Expected: PASS.

### Task 6: Clean repository outputs and finalize GitHub readiness

**Files:**
- Modify: `.gitignore`
- Delete: `src/robot_memory_vla.egg-info/*` from tracked source if appropriate
- Delete: `__pycache__` artifacts from tracked source if appropriate
- Modify: `README.md`
- Modify: `explain.md`
- Test: `tests/test_config.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Remove tracked local build artifacts**

Delete repo-owned generated files that should not be committed:
- `egg-info`
- `__pycache__`
- transient outputs not needed for source distribution

- [ ] **Step 2: Update `.gitignore`**

Ignore:
- Python caches
- build artifacts
- local downloaded model assets where appropriate
- runtime outputs under `data/`

- [ ] **Step 3: Run the final verification set**

Run:

```bash
python -m compileall -q src tests
pytest -q
python -m robot_memory_vla.app.main --preflight --config-dir ./configs
```

Expected:
- compileall succeeds
- full test suite passes
- preflight prints either `Preflight OK` or only repo-local missing asset guidance expected for fresh clones

- [ ] **Step 4: Final doc pass**

Confirm `README.md` only contains operational information and `explain.md` contains project explanation and architecture context.

- [ ] **Step 5: Produce GitHub handoff checklist**

Document a final checklist in `README.md` for fresh users:
- clone
- bootstrap
- download assets
- run tests
- run local demo
- optional real robot mode

