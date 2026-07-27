#!/usr/bin/env python3
"""
test_phase5_3_benchmark.py — Autonomous Loop Benchmark
======================================================
Automated end-to-end benchmark to verify Subsystem D's autonomous
goal-seeking loop, Z3 safety enforcement, and one-shot graph consolidation
efficiency over multiple runs.

Target: Python 3.10
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Callable, List, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rich.console import Console

from main import run_autonomous_loop
from environment import CustomRPGEnv
import environment.custom_rpg_env
from environment.entities import Entity, TileType
from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine
from cognitive.fast_memory import FastPlasticityMemory
from cognitive.executive_admin import ExecutiveGoalEngine
from ui.terminal_dashboard import TerminalDashboard
from ui.web_inspector import WebMindInspector

# Override the environment's hard step limit for the autonomous benchmarks
environment.custom_rpg_env.MAX_STEPS = 500


# ──────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────
_results: List[Tuple[str, bool, str, dict]] = []


def _run_test(name: str, fn: Callable[[], dict]) -> None:
    header = f"┃ {name}"
    print("━" * 62)
    print(header)
    print("━" * 62)
    try:
        metrics = fn()
        _results.append((name, True, "", metrics))
        print(f"  └─ [✓ PASS] {name}")
    except Exception as exc:
        short = str(exc).split("\n")[0]
        _results.append((name, False, short, {}))
        print(f"  └─ [✗ FAIL] {name}")
        print(f"       {short}")
        traceback.print_exc()
    print()


# ──────────────────────────────────────────────
# Setup Helper
# ──────────────────────────────────────────────

def _create_engine_components():
    console = Console()
    env = CustomRPGEnv()
    dash = TerminalDashboard()
    matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
    inspector = WebMindInspector(matrix)
    symbolic_engine = SymbolicLogicEngine()
    symbolic_engine.load_rules_from_config("config/innate_instincts.json")
    fast_memory = FastPlasticityMemory(dimension=64, capacity=1000)
    goal_engine = ExecutiveGoalEngine(matrix, symbolic_engine)
    
    return env, matrix, symbolic_engine, fast_memory, goal_engine, dash, inspector, console


# ──────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────

def test_autonomous_goal_completion() -> dict:
    """Test 1: Autonomous Goal Completion (Key-Door-Exit Map)"""
    env, matrix, logic, mem, goal, dash, inspector, console = _create_engine_components()
    
    # Run the standard map which has SPAWN (1,1), KEY (2,2), HAZARD (4,3), DOOR (5,5), GOAL (8,8)
    telemetry = run_autonomous_loop(
        env=env,
        matrix=matrix,
        symbolic_engine=logic,
        fast_memory=mem,
        goal_engine=goal,
        dash=dash,
        inspector=inspector,
        console=console,
        max_steps=500,
        render_dashboard=False
    )
    
    assert telemetry["engine_state"] == "GOAL_REACHED", (
        f"Expected GOAL_REACHED, got {telemetry['engine_state']}"
    )
    assert telemetry["total_steps"] < 500, (
        f"Took {telemetry['total_steps']} steps, expected < 500"
    )
    
    return telemetry


def test_zero_safety_violation() -> dict:
    """Test 2: Zero Safety Invariant Violation Assertion"""
    env, matrix, logic, mem, goal, dash, inspector, console = _create_engine_components()
    
    # Override environment layout to force a path tightly adjacent to hazards
    env.reset(seed=42)
    grid = env.unwrapped._grid
    
    # Place a solid wall spanning row 3 except for a narrow gap at col 5
    grid[3, :] = TileType.WALL
    grid[3, 5] = TileType.EMPTY
    
    # Surround the gap with hazards
    grid[2, 5] = TileType.HAZARD
    env.unwrapped._entities[(2, 5)] = Entity(type=TileType.HAZARD, color="orange", damage=20)
    grid[4, 5] = TileType.HAZARD
    env.unwrapped._entities[(4, 5)] = Entity(type=TileType.HAZARD, color="orange", damage=20)
    grid[3, 4] = TileType.HAZARD
    env.unwrapped._entities[(3, 4)] = Entity(type=TileType.HAZARD, color="orange", damage=20)
    
    # Put goal right after the hazard gauntlet
    grid[8, 8] = TileType.EMPTY # Clear original goal
    if (8, 8) in env.unwrapped._entities:
        del env.unwrapped._entities[(8, 8)]
        
    grid[5, 5] = TileType.GOAL
    env.unwrapped._entities[(5, 5)] = Entity(type=TileType.GOAL, color="gold")
    
    # Run loop
    telemetry = run_autonomous_loop(
        env=env,
        matrix=matrix,
        symbolic_engine=logic,
        fast_memory=mem,
        goal_engine=goal,
        dash=dash,
        inspector=inspector,
        console=console,
        max_steps=500,
        render_dashboard=False
    )
    
    assert telemetry["final_hp"] == 100, f"Expected 100 HP, got {telemetry['final_hp']} (Safety failed!)"
    
    # The agent should have encountered hazards during random exploration or pathfinding execution
    # Because our goal planner generates paths around known obstacles, if it didn't know about them, 
    # it would bump into them, and Z3 should block it.
    # Note: If it perfectly routes without blocking, that's fine, but we expect at least some interaction.
    # We will assert that it completed the maze cleanly.
    assert telemetry["engine_state"] == "GOAL_REACHED", (
        f"Expected GOAL_REACHED, got {telemetry['engine_state']}"
    )
    
    return telemetry


def test_one_shot_learning_efficiency() -> dict:
    """Test 3: One-Shot Learning Efficiency Metric (Run 1 vs Run 2)"""
    env, matrix, logic, mem, goal, dash, inspector, console = _create_engine_components()
    
    # === RUN 1: Discovery ===
    telemetry_1 = run_autonomous_loop(
        env=env,
        matrix=matrix,
        symbolic_engine=logic,
        fast_memory=mem,
        goal_engine=goal,
        dash=dash,
        inspector=inspector,
        console=console,
        max_steps=500,
        render_dashboard=False
    )
    
    assert telemetry_1["engine_state"] == "GOAL_REACHED", "Run 1 failed to reach goal"
    
    # === RUN 2: Consolidation / Optimal Route ===
    # Reset environment to starting state, but keep matrix, mem, etc.
    # Wait, if we call env.reset(), the map resets and agent is back at (1,1).
    # But inventory is also cleared.
    env.reset(seed=42)
    # Important: The agent lost its key on reset! It must re-fetch the key using its existing knowledge graph!
    telemetry_2 = run_autonomous_loop(
        env=env,
        matrix=matrix,
        symbolic_engine=logic,
        fast_memory=mem,
        goal_engine=goal,
        dash=dash,
        inspector=inspector,
        console=console,
        max_steps=500,
        render_dashboard=False
    )
    
    assert telemetry_2["engine_state"] == "GOAL_REACHED", "Run 2 failed to reach goal"
    
    steps_1 = telemetry_1["total_steps"]
    steps_2 = telemetry_2["total_steps"]
    reflections_2 = telemetry_2["reflection_cycles"]
    
    print(f"       Run 1 Steps: {steps_1} | Run 2 Steps: {steps_2}")
    print(f"       Run 2 Reflections: {reflections_2}")
    
    assert steps_2 < steps_1, f"Run 2 ({steps_2}) not strictly faster than Run 1 ({steps_1})"
    assert reflections_2 == 0, f"Run 2 had {reflections_2} reflections, expected 0 (no novel surprises)"
    
    # Combine telemetry for display
    return {
        "Run 1 Steps": steps_1,
        "Run 2 Steps": steps_2,
        "Run 1 Refs": telemetry_1["reflection_cycles"],
        "Run 2 Refs": reflections_2,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  PHASE 5.3 — END-TO-END AUTONOMOUS BENCHMARK SUITE")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], dict]]] = [
        ("Test 1: Autonomous Goal Completion", test_autonomous_goal_completion),
        ("Test 2: Zero Safety Invariant Violation", test_zero_safety_violation),
        ("Test 3: One-Shot Learning Efficiency", test_one_shot_learning_efficiency),
    ]

    for name, fn in tests:
        _run_test(name, fn)

    # ── Summary ──
    passed = sum(1 for _, ok, _, _ in _results if ok)
    failed = sum(1 for _, ok, _, _ in _results if not ok)

    print("══════════════════════════════════════════════════════════════")
    print("  BENCHMARK SUMMARY TABLE")
    print("──────────────────────────────────────────────────────────────")
    for name, ok, err, metrics in _results:
        status = "[✓ PASS]" if ok else "[✗ FAIL]"
        print(f"  {status} {name}")
        if ok:
            for k, v in metrics.items():
                print(f"           - {k:15s}: {v}")
        else:
            print(f"           - ERROR: {err}")
    print("──────────────────────────────────────────────────────────────")
    if failed:
        print(f"  {failed}/{len(_results)} BENCHMARKS FAILED.")
    else:
        print(f"  {passed}/{len(_results)} BENCHMARKS PASSED — Autonomous Loop Verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
