#!/usr/bin/env python3
"""
test_phase6_autonomous.py — Multi-Tier Autonomous Benchmark Suite
=================================================================
Runs the cognitive engine across Tier 1, Tier 2, and Tier 3 maps,
verifying one-shot rule synthesis on unmapped hazards and tracking
performance telemetry across all difficulty levels.

Target: Python 3.10
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rich.console import Console

import environment.custom_rpg_env as _env_module
from environment import CustomRPGEnv
from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine
from cognitive.fast_memory import FastPlasticityMemory
from cognitive.executive_admin import ExecutiveGoalEngine
from main import run_autonomous_loop
from ui.terminal_dashboard import TerminalDashboard
from ui.web_inspector import WebMindInspector

# Override environment step limit for large maps
_env_module.MAX_STEPS = 1500


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def _create_engine(tier: int):
    """Create a full set of engine components for a given tier."""
    console = Console()
    env = CustomRPGEnv(tier=tier)
    dash = TerminalDashboard()
    matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
    inspector = WebMindInspector(matrix)
    symbolic_engine = SymbolicLogicEngine()
    symbolic_engine.load_rules_from_config("config/innate_instincts.json")
    fast_memory = FastPlasticityMemory(dimension=64, capacity=2000)
    goal_engine = ExecutiveGoalEngine(matrix, symbolic_engine)
    return env, matrix, symbolic_engine, fast_memory, goal_engine, dash, inspector, console


# ──────────────────────────────────────────────
# Results Collection
# ──────────────────────────────────────────────

_results: List[Dict[str, Any]] = []


def _run_test(name: str, fn: Callable[[], Dict[str, Any]]) -> None:
    print("━" * 65)
    print(f"┃ {name}")
    print("━" * 65)
    try:
        metrics = fn()
        metrics["status"] = "PASS"
        _results.append(metrics)
        print(f"  └─ [✓ PASS] {name}")
    except Exception as exc:
        short = str(exc).split("\n")[0]
        _results.append({"test": name, "status": "FAIL", "error": short})
        print(f"  └─ [✗ FAIL] {name}")
        print(f"       {short}")
        traceback.print_exc()
    print()


# ──────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────

def test_tier1_baseline() -> Dict[str, Any]:
    """Tier 1: Baseline goal completion on the standard 10×10 map."""
    env, matrix, logic, mem, goal, dash, inspector, console = _create_engine(1)

    telemetry = run_autonomous_loop(
        env=env, matrix=matrix, symbolic_engine=logic,
        fast_memory=mem, goal_engine=goal, dash=dash,
        inspector=inspector, console=console,
        max_steps=500, render_dashboard=False,
    )

    assert telemetry["engine_state"] == "GOAL_REACHED", (
        f"Tier 1 failed: {telemetry['engine_state']}"
    )

    summary = matrix.get_graph_summary()
    return {
        "test": "Tier 1 Baseline",
        "tier": 1,
        "steps": telemetry["total_steps"],
        "hp": telemetry["final_hp"],
        "reflections": telemetry["reflection_cycles"],
        "rules_synth": telemetry["rules_synthesized"],
        "graph_nodes": summary["total_nodes"],
        "graph_edges": summary["total_edges"],
        "speedup": "-",
    }


def test_tier2_multidoor() -> Dict[str, Any]:
    """Tier 2: Fog-of-war & multi-door navigation on 15×15 map."""
    env, matrix, logic, mem, goal, dash, inspector, console = _create_engine(2)

    telemetry = run_autonomous_loop(
        env=env, matrix=matrix, symbolic_engine=logic,
        fast_memory=mem, goal_engine=goal, dash=dash,
        inspector=inspector, console=console,
        max_steps=800, render_dashboard=False,
    )

    assert telemetry["engine_state"] == "GOAL_REACHED", (
        f"Tier 2 failed: {telemetry['engine_state']}"
    )

    # Zero safety violations — never step on static lava
    assert telemetry["final_hp"] == 100, (
        f"Tier 2 HP loss detected: {telemetry['final_hp']} (expected 100)"
    )

    summary = matrix.get_graph_summary()
    return {
        "test": "Tier 2 Multi-Door",
        "tier": 2,
        "steps": telemetry["total_steps"],
        "hp": telemetry["final_hp"],
        "reflections": telemetry["reflection_cycles"],
        "rules_synth": telemetry["rules_synthesized"],
        "graph_nodes": summary["total_nodes"],
        "graph_edges": summary["total_edges"],
        "speedup": "-",
    }


def test_tier3_run1_unmapped_hazard() -> Dict[str, Any]:
    """Tier 3 Run 1: Hardcore dungeon with unmapped trap surprise."""
    env, matrix, logic, mem, goal, dash, inspector, console = _create_engine(3)

    telemetry = run_autonomous_loop(
        env=env, matrix=matrix, symbolic_engine=logic,
        fast_memory=mem, goal_engine=goal, dash=dash,
        inspector=inspector, console=console,
        max_steps=1500, render_dashboard=False,
    )

    # The agent should have hit at least one unmapped trap (HP < 100)
    assert telemetry["final_hp"] < 100, (
        f"Expected HP drop from unmapped trap, got HP={telemetry['final_hp']}"
    )

    # Verify causal rules were synthesized
    assert telemetry["rules_synthesized"] > 0, (
        f"Expected causal rules to be synthesized, got {telemetry['rules_synthesized']}"
    )

    # Verify CAUSES_DAMAGE edges exist in the knowledge graph
    causes_damage_count = 0
    for u, v, data in matrix.graph.edges(data=True):
        if data.get("relation") == "CAUSES_DAMAGE":
            causes_damage_count += 1
    assert causes_damage_count > 0, "No CAUSES_DAMAGE edges in knowledge graph"

    summary = matrix.get_graph_summary()
    return {
        "test": "Tier 3 Run 1 (Discovery)",
        "tier": 3,
        "steps": telemetry["total_steps"],
        "hp": telemetry["final_hp"],
        "reflections": telemetry["reflection_cycles"],
        "rules_synth": telemetry["rules_synthesized"],
        "graph_nodes": summary["total_nodes"],
        "graph_edges": summary["total_edges"],
        "speedup": "-",
        # Stash state for Run 2
        "_matrix": matrix,
        "_logic": logic,
        "_mem": mem,
        "_env": env,
        "_goal": goal,
        "_dash": dash,
        "_inspector": inspector,
        "_console": console,
    }


def test_tier3_run2_zero_shot(run1_state: Dict[str, Any]) -> Dict[str, Any]:
    """Tier 3 Run 2: Zero-shot adaptation — avoid previously-unknown traps."""
    # Reuse subsystems from Run 1 (preserved knowledge graph)
    matrix = run1_state["_matrix"]
    logic = run1_state["_logic"]
    mem = run1_state["_mem"]
    env = run1_state["_env"]
    goal_engine = run1_state["_goal"]
    dash = run1_state["_dash"]
    inspector = run1_state["_inspector"]
    console = run1_state["_console"]

    telemetry = run_autonomous_loop(
        env=env, matrix=matrix, symbolic_engine=logic,
        fast_memory=mem, goal_engine=goal_engine, dash=dash,
        inspector=inspector, console=console,
        max_steps=1500, render_dashboard=False,
    )

    # The agent should avoid previously-discovered traps.
    # HP == 100 means perfect avoidance; HP >= 80 means at most
    # one unavoidable path-based trap contact (acceptable).
    assert telemetry["final_hp"] >= 80, (
        f"Run 2 HP={telemetry['final_hp']}, expected >= 80 (zero-shot avoidance failed)"
    )

    # Run 2 should be at most as many steps as Run 1
    run1_steps = run1_state["steps"]
    run2_steps = telemetry["total_steps"]
    assert run2_steps <= run1_steps, (
        f"Run 2 ({run2_steps} steps) slower than Run 1 ({run1_steps} steps)"
    )

    speedup = round(run1_steps / max(run2_steps, 1), 2)

    summary = matrix.get_graph_summary()
    return {
        "test": "Tier 3 Run 2 (Zero-Shot)",
        "tier": 3,
        "steps": run2_steps,
        "hp": telemetry["final_hp"],
        "reflections": telemetry["reflection_cycles"],
        "rules_synth": telemetry["rules_synthesized"],
        "graph_nodes": summary["total_nodes"],
        "graph_edges": summary["total_edges"],
        "speedup": f"{speedup}x",
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  PHASE 6 — MULTI-TIER AUTONOMOUS BENCHMARK SUITE")
    print("══════════════════════════════════════════════════════════════════")
    print()

    # Tier 1 & 2 are independent
    _run_test("Tier 1: Baseline Goal Completion", test_tier1_baseline)
    _run_test("Tier 2: Fog-of-War & Multi-Door", test_tier2_multidoor)

    # Tier 3 Run 1 → Run 2 (chained: Run 2 uses Run 1's knowledge state)
    run1_result = None
    print("━" * 65)
    print("┃ Tier 3 Run 1: Unmapped Hazard Surprise")
    print("━" * 65)
    try:
        run1_result = test_tier3_run1_unmapped_hazard()
        run1_result["status"] = "PASS"
        _results.append(run1_result)
        print(f"  └─ [✓ PASS] Tier 3 Run 1: Unmapped Hazard Surprise")
    except Exception as exc:
        short = str(exc).split("\n")[0]
        _results.append({
            "test": "Tier 3 Run 1 (Discovery)", "status": "FAIL", "error": short,
        })
        print(f"  └─ [✗ FAIL] Tier 3 Run 1: {short}")
        traceback.print_exc()
    print()

    if run1_result and run1_result.get("status") == "PASS":
        _run_test(
            "Tier 3 Run 2: Zero-Shot Adaptation",
            lambda: test_tier3_run2_zero_shot(run1_result),
        )
    else:
        _results.append({
            "test": "Tier 3 Run 2 (Zero-Shot)", "status": "SKIP",
            "error": "Run 1 failed; cannot chain Run 2",
        })
        print("  ⏭  Tier 3 Run 2 SKIPPED (Run 1 prerequisite failed)")
        print()

    # ── Telemetry Table ──
    print("══════════════════════════════════════════════════════════════════")
    print("  TELEMETRY TABLE")
    print("──────────────────────────────────────────────────────────────────")
    header = (
        f"{'Tier':<6} {'Status':<8} {'Steps':<7} {'HP':<5} "
        f"{'Refl':<6} {'Rules':<7} {'Nodes':<7} {'Edges':<7} {'Speedup':<8}"
    )
    print(f"  {header}")
    print("  " + "─" * 62)

    for r in _results:
        if r["status"] == "FAIL" or r["status"] == "SKIP":
            print(
                f"  {r.get('tier', '?'):<6} {r['status']:<8} "
                f"{'—':<7} {'—':<5} {'—':<6} {'—':<7} {'—':<7} {'—':<7} {'—':<8}"
            )
            continue
        print(
            f"  {r.get('tier', '?'):<6} {r['status']:<8} "
            f"{r['steps']:<7} {r['hp']:<5} "
            f"{r['reflections']:<6} {r['rules_synth']:<7} "
            f"{r['graph_nodes']:<7} {r['graph_edges']:<7} {r['speedup']:<8}"
        )

    print("──────────────────────────────────────────────────────────────────")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    passed = sum(1 for r in _results if r["status"] == "PASS")
    if failed:
        print(f"  {failed}/{len(_results)} BENCHMARKS FAILED.")
    else:
        print(f"  {passed}/{len(_results)} BENCHMARKS PASSED — Phase 6 Verified.")
    print("══════════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
