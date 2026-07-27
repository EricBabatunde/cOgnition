#!/usr/bin/env python3
"""
test_step5_2.py — Executive Goal Engine Unit Test
=================================================
Automated unit test to verify Subsystem D's symbolic sub-goal
stack synthesis, backward chaining precondition decomposition,
and high-level execution plan compilation.

Target: Python 3.10
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Callable, List, Tuple

# Ensure project root is on sys.path when run from tests/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine
from cognitive.executive_admin import ExecutiveGoalEngine, GoalType


# ──────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────
_results: List[Tuple[str, bool, str]] = []


def _run_test(name: str, fn: Callable[[], None]) -> None:
    header = f"┃ {name}"
    print("━" * 62)
    print(header)
    print("━" * 62)
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  └─ [✓ PASS] {name}")
    except Exception as exc:
        short = str(exc).split("\n")[0]
        _results.append((name, False, short))
        print(f"  └─ [✗ FAIL] {name}")
        print(f"       {short}")
        traceback.print_exc()
    print()


# ──────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────

def test_scenario_1_direct_goal_path() -> None:
    """Scenario 1: Direct path to GOAL with no obstacles."""
    matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
    logic = SymbolicLogicEngine()
    engine = ExecutiveGoalEngine(matrix, logic)

    # 3x3 grid
    for r in range(3):
        for c in range(3):
            matrix.add_spatial_node(r, c, "EMPTY")
            # Connect row elements
            if c > 0:
                matrix.add_typed_edge(f"Tile_{r}_{c-1}", f"Tile_{r}_{c}", "CONNECTS_TO")
                matrix.add_typed_edge(f"Tile_{r}_{c}", f"Tile_{r}_{c-1}", "CONNECTS_TO")
            # Connect col elements
            if r > 0:
                matrix.add_typed_edge(f"Tile_{r-1}_{c}", f"Tile_{r}_{c}", "CONNECTS_TO")
                matrix.add_typed_edge(f"Tile_{r}_{c}", f"Tile_{r-1}_{c}", "CONNECTS_TO")

    # Add GOAL at (2, 0)
    matrix.add_entity_node("goal_exit", "GOAL", {"location": "(2,0)"})

    goals = engine.synthesize_goal_stack((0, 0), [])
    
    assert len(goals) == 1, f"Expected exactly 1 goal, got {len(goals)}"
    assert goals[0].goal_type == GoalType.REACH_EXIT, f"Expected REACH_EXIT, got {goals[0].goal_type}"
    assert goals[0].target_pos == (2, 0), f"Expected (2,0), got {goals[0].target_pos}"

    print(f"       Synthesized Goals: {[g.goal_type.name for g in goals]}")


def _build_scenario_2_matrix() -> CoreKnowledgeMatrix:
    """Build a 5x5 graph with DOOR at (2,0), EXIT at (4,0), KEY at (0,2).
    A solid wall spans row 2, except for the door at (2,0)."""
    matrix = CoreKnowledgeMatrix("config/innate_instincts.json")

    for r in range(5):
        for c in range(5):
            if (r, c) == (2, 0):
                matrix.add_spatial_node(r, c, "DOOR")
            elif r == 2:
                matrix.add_spatial_node(r, c, "WALL")
            else:
                matrix.add_spatial_node(r, c, "EMPTY")

            # Connect row elements
            if c > 0:
                matrix.add_typed_edge(f"Tile_{r}_{c-1}", f"Tile_{r}_{c}", "CONNECTS_TO")
                matrix.add_typed_edge(f"Tile_{r}_{c}", f"Tile_{r}_{c-1}", "CONNECTS_TO")
            # Connect col elements
            if r > 0:
                matrix.add_typed_edge(f"Tile_{r-1}_{c}", f"Tile_{r}_{c}", "CONNECTS_TO")
                matrix.add_typed_edge(f"Tile_{r}_{c}", f"Tile_{r-1}_{c}", "CONNECTS_TO")

    matrix.add_entity_node("door_main", "DOOR", {"location": "(2,0)"})
    matrix.add_entity_node("goal_exit", "GOAL", {"location": "(4,0)"})
    matrix.add_entity_node("key_red", "KEY", {"location": "(0,2)"})
    
    return matrix


def test_scenario_2_backward_chaining() -> None:
    """Scenario 2: Path blocked by locked door, requiring key fetch."""
    matrix = _build_scenario_2_matrix()
    logic = SymbolicLogicEngine()
    engine = ExecutiveGoalEngine(matrix, logic)

    goals = engine.synthesize_goal_stack((0, 0), [])
    
    expected_types = [GoalType.FETCH_KEY, GoalType.UNLOCK_DOOR, GoalType.REACH_EXIT]
    actual_types = [g.goal_type for g in goals]
    assert actual_types == expected_types, f"Expected {expected_types}, got {actual_types}"

    assert goals[0].target_pos == (0, 2), "FETCH_KEY should target (0,2)"
    assert goals[1].target_pos == (2, 0), "UNLOCK_DOOR should target (2,0)"
    assert goals[2].target_pos == (4, 0), "REACH_EXIT should target (4,0)"

    print(f"       Synthesized Goals: {[g.goal_type.name for g in goals]}")


def test_scenario_3_execution_plan_compilation() -> None:
    """Scenario 3: Compile multi-step sub-goal stack into actions."""
    matrix = _build_scenario_2_matrix()
    logic = SymbolicLogicEngine()
    engine = ExecutiveGoalEngine(matrix, logic)

    goals = engine.synthesize_goal_stack((0, 0), [])
    # Start (0,0) facing EAST (0,1)
    plan = engine.compile_execution_plan(goals, (0, 0), (0, 1), [])

    print(f"       Action Plan Length: {len(plan)}")
    print(f"       Plan: {plan}")

    # The plan must contain PICK_UP for the key and TOGGLE_INTERACT for the door
    assert "PICK_UP" in plan, "Plan must contain PICK_UP action"
    assert "TOGGLE_INTERACT" in plan, "Plan must contain TOGGLE_INTERACT action"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STEP 5.2 — GOAL SYNTHESIS & PLAN COMPILATION VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Scenario 1 — Direct Goal Path", test_scenario_1_direct_goal_path),
        ("Scenario 2 — Backward Chaining Key/Door Decomposition", test_scenario_2_backward_chaining),
        ("Scenario 3 — Execution Plan Compilation", test_scenario_3_execution_plan_compilation),
    ]

    for name, fn in tests:
        _run_test(name, fn)

    # ── Summary ──
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    print("══════════════════════════════════════════════════════════════")
    print("  TEST SUMMARY")
    print("──────────────────────────────────────────────────────────────")
    for name, ok, err in _results:
        status = "[✓ PASS]" if ok else "[✗ FAIL]"
        line = f"  {status}  {name}"
        if err:
            line += f"  ({err})"
        print(line)
    print("──────────────────────────────────────────────────────────────")
    if failed:
        print(f"  {failed}/{len(_results)} TESTS FAILED — Review errors above.")
    else:
        print(f"  {passed}/{len(_results)} TESTS PASSED — Goal engine verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
