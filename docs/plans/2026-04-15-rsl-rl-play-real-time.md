# RSL-RL Play Real-Time Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional `--real-time` flag to `TextOpTracker/scripts/rsl_rl/play.py` so interactive playback can run close to the environment control timestep instead of always running at maximum speed.

**Architecture:** Keep the current script structure intact and make a minimal change inside the existing play loop. Add a small pacing helper inside `play.py`, wire it to a new CLI flag, and use AST-based tests because the script currently has top-level simulator startup side effects that make normal imports impractical in unit tests.

**Tech Stack:** Python, argparse, AST-based pytest tests, `py_compile`

---

### Task 1: Add failing tests for the new CLI flag and pacing helper

**Files:**
- Create: `TextOpTracker/tests/test_rsl_rl_play.py`
- Modify: `TextOpTracker/scripts/rsl_rl/play.py`

**Step 1: Write the failing test**

Add tests that:
- assert `play.py` registers a `--real-time` CLI flag with `action="store_true"` and `default=False`
- extract a new `_sleep_to_real_time(...)` helper from the script and verify it sleeps for the remaining positive timestep only
- assert the script calls `_sleep_to_real_time(...)` with `args_cli.real_time`

**Step 2: Run test to verify it fails**

Run: `pytest TextOpTracker/tests/test_rsl_rl_play.py -q`

Expected: FAIL because `--real-time` and the pacing helper do not exist yet.

### Task 2: Implement the minimal real-time pacing change

**Files:**
- Modify: `TextOpTracker/scripts/rsl_rl/play.py`
- Test: `TextOpTracker/tests/test_rsl_rl_play.py`

**Step 1: Write minimal implementation**

Implement:
- `import time`
- `parser.add_argument("--real-time", action="store_true", default=False, help="Run close to real-time if possible.")`
- `_sleep_to_real_time(real_time, step_dt, step_start_time, *, now_fn=time.time, sleep_fn=time.sleep)`
- a call in the play loop that records the step start time, performs policy inference and `env.step(actions)`, then invokes `_sleep_to_real_time(...)`

**Step 2: Run test to verify it passes**

Run: `pytest TextOpTracker/tests/test_rsl_rl_play.py -q`

Expected: PASS

### Task 3: Verify the script still parses cleanly

**Files:**
- Modify: `TextOpTracker/scripts/rsl_rl/play.py`

**Step 1: Run focused verification**

Run: `python -m py_compile TextOpTracker/scripts/rsl_rl/play.py`

Expected: no output

**Step 2: Re-run tests**

Run: `pytest TextOpTracker/tests/test_rsl_rl_play.py -q`

Expected: PASS
