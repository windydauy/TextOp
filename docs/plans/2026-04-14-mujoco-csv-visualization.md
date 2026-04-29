# MuJoCo CSV Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional MuJoCo playback window to `mujoco_csv_to_npz.py` so converted motions are visualized by default and skipped with `--headless`.

**Architecture:** Keep conversion and saving in the existing script, then call a new playback helper that reuses the converted root and joint arrays. Preserve a clean CLI by adding a single `--headless` switch and keep the visualization logic isolated in its own function for testing.

**Tech Stack:** Python, NumPy, MuJoCo viewer, pytest-style unit tests

---

### Task 1: Add failing tests for CLI visualization control

**Files:**
- Create: `TextOpTracker/tests/test_mujoco_csv_to_npz.py`
- Modify: `TextOpTracker/scripts/mujoco_csv_to_npz.py`

**Step 1: Write the failing tests**

Add tests that:
- verify `--headless` defaults to `False` and flips to `True` when passed
- verify `main()` skips visualization when `--headless` is set
- verify `main()` triggers visualization by default

**Step 2: Run test to verify it fails**

Run: `pytest TextOpTracker/tests/test_mujoco_csv_to_npz.py -q`

Expected: FAIL because `--headless` and visualization control do not exist yet.

### Task 2: Add minimal conversion result and playback helper

**Files:**
- Modify: `TextOpTracker/scripts/mujoco_csv_to_npz.py`
- Test: `TextOpTracker/tests/test_mujoco_csv_to_npz.py`

**Step 1: Write the failing test**

Add a test for a new playback helper that replays all converted frames into a fake MuJoCo viewer by updating root pose and joint state frame-by-frame.

**Step 2: Run test to verify it fails**

Run: `pytest TextOpTracker/tests/test_mujoco_csv_to_npz.py -q`

Expected: FAIL because playback helper and returned conversion payload do not exist yet.

**Step 3: Write minimal implementation**

Implement:
- a small dataclass to carry the converted motion arrays and output path
- a playback helper that opens `mujoco.viewer.launch_passive(...)`
- a small helper for joint qpos/dof address lookup
- CLI wiring in `main()` so playback runs unless `--headless`

**Step 4: Run test to verify it passes**

Run: `pytest TextOpTracker/tests/test_mujoco_csv_to_npz.py -q`

Expected: PASS

### Task 3: Verify the script still parses and saves output

**Files:**
- Modify: `TextOpTracker/scripts/mujoco_csv_to_npz.py`

**Step 1: Run focused verification**

Run: `python -m py_compile TextOpTracker/scripts/mujoco_csv_to_npz.py`

Expected: no output

**Step 2: Run tests again**

Run: `pytest TextOpTracker/tests/test_mujoco_csv_to_npz.py -q`

Expected: PASS
