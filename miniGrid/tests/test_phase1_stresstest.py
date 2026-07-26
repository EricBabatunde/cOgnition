#!/usr/bin/env python3
"""
test_phase1_stresstest.py — Phase 1 Headless Stress Test
=========================================================
Executes 1,000 randomly-sampled actions across multiple
environment episodes to verify spatial invariants, observation
shapes, and health bounds hold under sustained random load.

Target: Python 3.10
"""

from __future__ import annotations

import random
import time
from typing import List, Tuple

from environment import Action, CustomRPGEnv

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TOTAL_STEPS: int = 1_000
GRID_MIN: int = 1          # Inner boundary (row/col min)
GRID_MAX: int = 8          # Inner boundary (row/col max)
HP_MIN: int = 0
HP_MAX: int = 100
FOV_SHAPE: Tuple[int, int] = (5, 5)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _hp_bar(hp: int) -> str:
    """Return a coloured HP label for terminal output."""
    if hp > 60:
        return f"\033[92mHP={hp}\033[0m"
    if hp > 30:
        return f"\033[93mHP={hp}\033[0m"
    return f"\033[91mHP={hp}\033[0m"


def main() -> None:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  PHASE 1 — HEADLESS ENVIRONMENT STRESS TEST  (1,000 steps)")
    print("══════════════════════════════════════════════════════════════")
    print()

    env = CustomRPGEnv()
    obs, info = env.reset(seed=42)

    episode: int = 1
    episode_steps: int = 0
    total_resets: int = 0
    episode_log: List[str] = []

    t_start = time.perf_counter()

    for step_idx in range(1, TOTAL_STEPS + 1):
        # ── Sample random action ──
        action = env.action_space.sample()

        # ── Step environment ──
        obs, reward, terminated, truncated, info = env.step(action)
        ps = obs["player_state"]
        episode_steps += 1

        pos = ps["position"]
        hp = ps["health"]
        fov = obs["fov"]

        # ══════════════════════════════════════════════
        #  INVARIANT ASSERTIONS (every single step)
        # ══════════════════════════════════════════════

        # 1. Spatial boundary: player must remain inside walls
        assert GRID_MIN <= pos[0] <= GRID_MAX, (
            f"Step {step_idx}: Row {pos[0]} out of bounds "
            f"[{GRID_MIN}, {GRID_MAX}]"
        )
        assert GRID_MIN <= pos[1] <= GRID_MAX, (
            f"Step {step_idx}: Col {pos[1]} out of bounds "
            f"[{GRID_MIN}, {GRID_MAX}]"
        )

        # 2. Health bounds
        assert HP_MIN <= hp <= HP_MAX, (
            f"Step {step_idx}: HP={hp} out of bounds "
            f"[{HP_MIN}, {HP_MAX}]"
        )

        # 3. FOV shape
        assert fov.shape == FOV_SHAPE, (
            f"Step {step_idx}: FOV shape {fov.shape} != {FOV_SHAPE}"
        )

        # ── Episode termination handling ──
        if terminated or truncated:
            if hp <= 0:
                outcome = f"💀 HP depleted at step {episode_steps}"
            elif terminated:
                outcome = f"🏆 Goal reached at step {episode_steps}"
            else:
                outcome = f"⏱  Truncated at step {episode_steps}"

            episode_log.append(
                f"  Episode {episode:3d}: {outcome}  "
                f"({_hp_bar(hp)}, explored={info['explored_pct']*100:.0f}%)"
            )

            # Reset for next episode
            obs, info = env.reset(seed=42 + episode)
            ps = obs["player_state"]
            episode += 1
            episode_steps = 0
            total_resets += 1

    t_elapsed = time.perf_counter() - t_start
    sps = TOTAL_STEPS / t_elapsed if t_elapsed > 0 else float("inf")

    # ──────────────────────────────────────────────
    #  Results
    # ──────────────────────────────────────────────
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("┃ Episode Log")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for entry in episode_log:
        print(entry)
    if not episode_log:
        print("  (no episodes terminated within 1,000 steps)")
    print()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("┃ Performance Telemetry")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Total Steps      : {TOTAL_STEPS}")
    print(f"  Episodes          : {episode} ({total_resets} resets)")
    print(f"  Elapsed Time      : {t_elapsed:.3f}s")
    print(f"  Steps Per Second  : {sps:,.0f} SPS")
    print()

    print("══════════════════════════════════════════════════════════════")
    print("  [✓ PASS]  1,000 steps — ZERO invariant violations")
    print("  [✓ PASS]  Spatial bounds [1..8] held on every step")
    print("  [✓ PASS]  Health bounds [0..100] held on every step")
    print("  [✓ PASS]  FOV shape (5,5) correct on every step")
    print("  [✓ PASS]  No unhandled exceptions")
    print("══════════════════════════════════════════════════════════════")
    print()


if __name__ == "__main__":
    main()
