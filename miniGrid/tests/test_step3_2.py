#!/usr/bin/env python3
"""
test_step3_2.py — Dynamic Rule Compiler Unit Verification
==========================================================
Automated unit test verifying dynamic JSON rule ingestion,
confidence-threshold filtering, and Z3 SMT evaluation in
``SymbolicLogicEngine``.

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

def test_json_rule_ingestion() -> None:
    """Test 1 — Load innate rules from JSON and verify count."""
    count = _sle.load_rules_from_config("config/innate_instincts.json")

    assert count == 4, (
        f"Expected 4 rules loaded, got {count}"
    )
    assert len(_sle.loaded_rules) == 4, (
        f"Expected 4 rules in loaded_rules, got {len(_sle.loaded_rules)}"
    )

    rule_ids = [r.rule_id for r in _sle.loaded_rules]
    for expected_id in ["wall_blocking", "goal_priority",
                        "locked_door_requires_key", "hazard_avoidance"]:
        assert expected_id in rule_ids, (
            f"Rule '{expected_id}' missing from loaded rules"
        )

    print(f"       Loaded {count} rules: {rule_ids}")


def test_confidence_threshold_filtering() -> None:
    """Test 2 — Rules below min_confidence are excluded from active trace."""
    # At threshold 0.92:
    #   wall_blocking       (1.0)  → ACTIVE
    #   goal_priority       (0.95) → ACTIVE
    #   locked_door_requires_key (0.90) → EXCLUDED
    #   hazard_avoidance    (1.0)  → ACTIVE
    safe, reason, status, active = _sle.verify_action_dynamic(
        action_name="MOVE_FORWARD",
        state_context={},
        min_confidence=0.92,
    )

    assert "locked_door_requires_key" not in active, (
        f"'locked_door_requires_key' (conf=0.90) should be excluded at "
        f"threshold 0.92, but was in active: {active}"
    )
    assert "wall_blocking" in active, (
        f"'wall_blocking' (conf=1.0) should be active at threshold 0.92"
    )
    assert "hazard_avoidance" in active, (
        f"'hazard_avoidance' (conf=1.0) should be active at threshold 0.92"
    )
    assert "goal_priority" in active, (
        f"'goal_priority' (conf=0.95) should be active at threshold 0.92"
    )

    print(f"       Threshold: 0.92")
    print(f"       Active rules:   {active}")
    print(f"       Excluded:       locked_door_requires_key (conf=0.90)")


def test_dynamic_wall_blocking() -> None:
    """Test 3 — MOVE_FORWARD into wall must be UNSAT with wall_blocking active."""
    safe, reason, status, active = _sle.verify_action_dynamic(
        action_name="MOVE_FORWARD",
        state_context={"is_wall": True},
    )

    assert safe is False, (
        f"Expected safe=False for wall, got {safe}"
    )
    assert status == "UNSAT", (
        f"Expected UNSAT, got {status}"
    )
    assert "wall_blocking" in active, (
        f"'wall_blocking' should be in active trace, got {active}"
    )

    print(f"       Action:  MOVE_FORWARD → WALL")
    print(f"       Result:  safe={safe}  status={status}")
    print(f"       Active:  {active}")


def test_dynamic_locked_door_no_key() -> None:
    """Test 4 — MOVE_FORWARD into locked door without key must be UNSAT."""
    safe, reason, status, active = _sle.verify_action_dynamic(
        action_name="MOVE_FORWARD",
        state_context={
            "is_door": True,
            "is_locked": True,
            "has_key": False,
        },
    )

    assert safe is False, (
        f"Expected safe=False for locked door without key, got {safe}"
    )
    assert status == "UNSAT", (
        f"Expected UNSAT, got {status}"
    )
    assert "locked_door_requires_key" in active, (
        f"'locked_door_requires_key' should be in active trace, got {active}"
    )

    print(f"       Action:  MOVE_FORWARD → DOOR (locked, no key)")
    print(f"       Result:  safe={safe}  status={status}")
    print(f"       Active:  {active}")


def test_dynamic_locked_door_with_key() -> None:
    """Test 5 — MOVE_FORWARD into locked door WITH key must be SAT."""
    safe, reason, status, active = _sle.verify_action_dynamic(
        action_name="MOVE_FORWARD",
        state_context={
            "is_door": True,
            "is_locked": True,
            "has_key": True,
        },
    )

    assert safe is True, (
        f"Expected safe=True for locked door with key, got {safe}"
    )
    assert status == "SAT", (
        f"Expected SAT, got {status}"
    )

    print(f"       Action:  MOVE_FORWARD → DOOR (locked, HAS key)")
    print(f"       Result:  safe={safe}  status={status}")
    print(f"       Active:  {active}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STEP 3.2 — DYNAMIC RULE COMPILER UNIT VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    _setup()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — JSON rule ingestion", test_json_rule_ingestion),
        ("Test 2 — Confidence threshold filtering", test_confidence_threshold_filtering),
        ("Test 3 — Dynamic SMT wall blocking", test_dynamic_wall_blocking),
        ("Test 4 — Dynamic SMT locked door (no key)", test_dynamic_locked_door_no_key),
        ("Test 5 — Dynamic SMT locked door (WITH key)", test_dynamic_locked_door_with_key),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Dynamic Rule Compiler verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
