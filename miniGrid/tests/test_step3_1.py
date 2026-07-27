#!/usr/bin/env python3
"""
test_step3_1.py — Symbolic Logic Engine Unit Verification
==========================================================
Automated unit test script verifying the mathematical safety
assertions produced by ``SymbolicLogicEngine`` across all four
first-order axioms using Z3 SAT/UNSAT outcomes.

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

from cognitive.symbolic_engine import SymbolicLogicEngine

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
# Shared fixture
# ──────────────────────────────────────────────
_sle: SymbolicLogicEngine | None = None


def _setup() -> None:
    global _sle
    _sle = SymbolicLogicEngine()


# ──────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────

def test_valid_movement_clear_tile() -> None:
    """Test 1 — MOVE_FORWARD into EMPTY tile must be SAT."""
    is_safe, reason, status = _sle.verify_action_safety(
        action_name="MOVE_FORWARD",
        target_tile_type="EMPTY",
        is_locked=False,
        inventory=[],
    )

    assert is_safe is True, (
        f"Expected safe=True for EMPTY tile, got {is_safe}"
    )
    assert status == "SAT", (
        f"Expected SAT, got {status}"
    )

    print(f"       Action: MOVE_FORWARD → EMPTY")
    print(f"       Result: safe={is_safe}  status={status}")
    print(f"       Reason: {reason}")


def test_wall_collision_prevention() -> None:
    """Test 2 — MOVE_FORWARD into WALL must be UNSAT (Axiom 1)."""
    is_safe, reason, status = _sle.verify_action_safety(
        action_name="MOVE_FORWARD",
        target_tile_type="WALL",
        is_locked=False,
        inventory=[],
    )

    assert is_safe is False, (
        f"Expected safe=False for WALL tile, got {is_safe}"
    )
    assert status == "UNSAT", (
        f"Expected UNSAT, got {status}"
    )

    print(f"       Action: MOVE_FORWARD → WALL")
    print(f"       Result: safe={is_safe}  status={status}")
    print(f"       Reason: {reason}")


def test_hazard_protection_invariant() -> None:
    """Test 3 — MOVE_FORWARD into HAZARD must be UNSAT (Axiom 2)."""
    is_safe, reason, status = _sle.verify_action_safety(
        action_name="MOVE_FORWARD",
        target_tile_type="HAZARD",
        is_locked=False,
        inventory=[],
    )

    assert is_safe is False, (
        f"Expected safe=False for HAZARD tile, got {is_safe}"
    )
    assert status == "UNSAT", (
        f"Expected UNSAT, got {status}"
    )

    print(f"       Action: MOVE_FORWARD → HAZARD")
    print(f"       Result: safe={is_safe}  status={status}")
    print(f"       Reason: {reason}")


def test_locked_door_without_key() -> None:
    """Test 4 — MOVE_FORWARD into locked DOOR without key must be UNSAT (Axiom 3)."""
    is_safe, reason, status = _sle.verify_action_safety(
        action_name="MOVE_FORWARD",
        target_tile_type="DOOR",
        is_locked=True,
        inventory=[],
    )

    assert is_safe is False, (
        f"Expected safe=False for locked DOOR without key, got {is_safe}"
    )
    assert status == "UNSAT", (
        f"Expected UNSAT, got {status}"
    )

    print(f"       Action: MOVE_FORWARD → DOOR (locked, no key)")
    print(f"       Result: safe={is_safe}  status={status}")
    print(f"       Reason: {reason}")


def test_locked_door_with_key() -> None:
    """Test 5 — MOVE_FORWARD into locked DOOR WITH key must be SAT (Axiom 3 satisfied)."""
    is_safe, reason, status = _sle.verify_action_safety(
        action_name="MOVE_FORWARD",
        target_tile_type="DOOR",
        is_locked=True,
        inventory=["KEY"],
    )

    assert is_safe is True, (
        f"Expected safe=True for locked DOOR with key, got {is_safe}"
    )
    assert status == "SAT", (
        f"Expected SAT, got {status}"
    )

    print(f"       Action: MOVE_FORWARD → DOOR (locked, HAS key)")
    print(f"       Result: safe={is_safe}  status={status}")
    print(f"       Reason: {reason}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STEP 3.1 — SYMBOLIC LOGIC ENGINE UNIT VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    _setup()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Valid movement into clear tile", test_valid_movement_clear_tile),
        ("Test 2 — Wall collision prevention (Axiom 1)", test_wall_collision_prevention),
        ("Test 3 — Hazard protection invariant (Axiom 2)", test_hazard_protection_invariant),
        ("Test 4 — Locked door without key (Axiom 3)", test_locked_door_without_key),
        ("Test 5 — Locked door WITH key (Axiom 3 SAT)", test_locked_door_with_key),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Symbolic Logic Engine verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
