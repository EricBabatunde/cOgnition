#!/usr/bin/env python3
"""
test_step5_1.py — Graph Pathfinding & Action Planner Unit Test
==============================================================
Automated unit test to verify A* topological pathfinding over
the CoreKnowledgeMatrix graph, obstacle bypass logic, locked
door inventory gating, and coordinate-to-action compilation.

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
# Shared graph builder
# ──────────────────────────────────────────────

def _build_5x5_matrix() -> CoreKnowledgeMatrix:
    """Build a synthetic 5×5 spatial graph with obstacles.

    Layout (row, col):
        (1, 1) = WALL
        (2, 1) = HAZARD
        (3, 1) = DOOR (locked)
        All others = EMPTY

    Bi-directional CONNECTS_TO edges between all adjacent tiles.
    """
    matrix = CoreKnowledgeMatrix("config/innate_instincts.json")

    for r in range(5):
        for c in range(5):
            if (r, c) == (1, 1):
                matrix.add_spatial_node(r, c, "WALL")
            elif (r, c) == (2, 1):
                matrix.add_spatial_node(r, c, "HAZARD")
            elif (r, c) == (3, 1):
                matrix.add_spatial_node(r, c, "DOOR")
            else:
                matrix.add_spatial_node(r, c, "EMPTY")

    # Bi-directional connectivity
    for r in range(5):
        for c in range(5):
            src = f"Tile_{r}_{c}"
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 5 and 0 <= nc < 5:
                    dst = f"Tile_{nr}_{nc}"
                    matrix.add_typed_edge(src, dst, "CONNECTS_TO")

    return matrix


# ──────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────

def test_obstacle_bypass_pathfinding() -> None:
    """Test 1 — Obstacle Bypass Pathfinding (WALL + HAZARD + locked DOOR)."""
    matrix = _build_5x5_matrix()

    path = matrix.find_topological_path(
        start_pos=(0, 1), target_pos=(4, 1), known_inventory=[]
    )

    assert path is not None, "Expected a valid path, got None"
    assert (1, 1) not in path, f"Path must not cross WALL at (1,1): {path}"
    assert (2, 1) not in path, f"Path must not cross HAZARD at (2,1): {path}"
    assert (3, 1) not in path, f"Path must not cross locked DOOR at (3,1): {path}"
    assert path[0] == (0, 1), f"Path must start at (0,1), got {path[0]}"
    assert path[-1] == (4, 1), f"Path must end at (4,1), got {path[-1]}"

    print(f"       Path: {path}")
    print(f"       Length: {len(path)} nodes")
    print(f"       Avoids: (1,1) WALL, (2,1) HAZARD, (3,1) locked DOOR")


def test_locked_door_traversal_with_key() -> None:
    """Test 2 — Locked Door Traversal with KEY in inventory."""
    matrix = _build_5x5_matrix()

    # Route from (2, 2) to (3, 0). With KEY, the shortest path runs
    # directly through the door: (2,2)→(3,2)→(3,1)→(3,0) = 4 hops.
    # Without KEY, must detour via row 4: much longer.
    path_with_key = matrix.find_topological_path(
        start_pos=(2, 2), target_pos=(3, 0), known_inventory=["KEY"]
    )
    path_without_key = matrix.find_topological_path(
        start_pos=(2, 2), target_pos=(3, 0), known_inventory=[]
    )

    assert path_with_key is not None, "Expected valid path through DOOR with KEY"
    assert (3, 1) in path_with_key, (
        f"With KEY, path should traverse DOOR at (3,1): {path_with_key}"
    )
    # Verify WALL and HAZARD are still blocked even with KEY
    assert (1, 1) not in path_with_key, f"WALL must still be blocked: {path_with_key}"
    assert (2, 1) not in path_with_key, f"HAZARD must still be blocked: {path_with_key}"

    # Without key, the path must avoid the door and be longer
    assert path_without_key is not None, "Should still find a detour path"
    assert (3, 1) not in path_without_key, (
        f"Without KEY, DOOR at (3,1) must be blocked: {path_without_key}"
    )
    assert len(path_without_key) > len(path_with_key), (
        f"Detour should be longer: with_key={len(path_with_key)}, "
        f"without_key={len(path_without_key)}"
    )

    print(f"       Path (with KEY): {path_with_key}")
    print(f"       Path (no KEY):   {path_without_key}")
    print(f"       Door (3,1) traversed with KEY: ✓")
    print(f"       Detour longer without KEY: ✓")


def test_action_sequence_compilation() -> None:
    """Test 3 — Action Sequence Compilation.

    Path: [(1,1), (1,2), (2,2)]
    Initial facing: EAST = (0, 1) in the direction ring.

    Direction ring (clockwise):
        idx 0 = NORTH (-1, 0)
        idx 1 = EAST  ( 0, 1)
        idx 2 = SOUTH ( 1, 0)
        idx 3 = WEST  ( 0,-1)

    Step 1: (1,1) → (1,2)  delta=(0,1)=EAST.  Facing EAST → MOVE_FORWARD.
    Step 2: (1,2) → (2,2)  delta=(1,0)=SOUTH.  Facing EAST, need SOUTH
            → 1 clockwise step → TURN_RIGHT, MOVE_FORWARD.

    Expected: ["MOVE_FORWARD", "TURN_RIGHT", "MOVE_FORWARD"]
    """
    matrix = _build_5x5_matrix()

    path = [(1, 1), (1, 2), (2, 2)]
    # EAST = (0, 1) in the clockwise direction ring
    actions = matrix.plan_action_sequence(path, current_direction=(0, 1))

    expected = ["MOVE_FORWARD", "TURN_RIGHT", "MOVE_FORWARD"]
    assert actions == expected, f"Expected {expected}, got {actions}"

    print(f"       Path: {path}")
    print(f"       Initial Facing: EAST (0, 1)")
    print(f"       Actions: {actions}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STEP 5.1 — GRAPH PATHFINDING & ACTION PLANNER VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Obstacle Bypass Pathfinding", test_obstacle_bypass_pathfinding),
        ("Test 2 — Locked Door Traversal with KEY", test_locked_door_traversal_with_key),
        ("Test 3 — Action Sequence Compilation", test_action_sequence_compilation),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Pathfinding verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
