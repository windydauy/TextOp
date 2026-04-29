# RSL-RL Play Real-Time Design

**Date:** 2026-04-15

**Goal:** Add an optional real-time playback mode to `TextOpTracker/scripts/rsl_rl/play.py` so policy inference can be viewed at approximately normal simulation speed instead of always running as fast as possible.

## Context

The current `play.py` loop steps the policy and environment continuously with no pacing. This is useful for throughput, but it makes interactive playback too fast to inspect visually.

The user wants an opt-in behavior that preserves the current default and only slows playback when requested from the CLI.

## Chosen Approach

Add a new `--real-time` boolean CLI flag to `play.py`. When enabled, each loop iteration records its start time, then sleeps after `env.step(actions)` long enough to match the environment control timestep from `env.unwrapped.step_dt`.

This mirrors the timing pattern already used in the provided IsaacLab example:
- measure per-step elapsed wall-clock time
- compute `step_dt - elapsed`
- only sleep when the remainder is positive

## Alternatives Considered

1. Always run in real time.
This is simpler, but it changes existing behavior for users who rely on maximum-speed playback.

2. Add a configurable `--fps` or `--speed` multiplier.
This is more flexible, but it adds API surface and complexity that the current request does not need.

3. Tie playback pacing to video recording only.
This would not solve the general interactive playback problem.

## Design Details

### CLI

Add:

```text
--real-time
```

Behavior:
- default `False`
- when omitted, playback remains unchanged and runs at maximum speed
- when provided, playback tries to match `env.unwrapped.step_dt`

### Runtime Pacing

Add a tiny helper in `play.py` for real-time pacing so timing behavior is isolated and testable.

Expected behavior:
- if real-time mode is disabled, return immediately
- if enabled, compute elapsed wall-clock time since the start of the loop iteration
- sleep only for the remaining positive duration
- if the step already took longer than `step_dt`, do not sleep

### Compatibility

The change should not alter:
- checkpoint loading
- motion file override logic
- video recording setup
- ONNX export behavior
- multi-agent to single-agent conversion

## Testing Strategy

`play.py` has heavy top-level simulator startup side effects, so direct module import is not a good unit-test target right now. Instead, add focused tests that:
- parse the script AST and assert `--real-time` is registered as a `store_true` flag
- extract and execute the new pacing helper to verify the sleep behavior
- assert the main loop calls the pacing helper with `args_cli.real_time`

## Risks

The main risk is unintentionally changing playback defaults or placing the sleep in the wrong part of the loop. Keeping the new logic behind an opt-in flag and using a small helper minimizes that risk.
