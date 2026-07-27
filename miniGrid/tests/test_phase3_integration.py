#!/usr/bin/env python3
"""
test_phase3_integration.py — Phase 3 Integration Verification
=============================================================
End-to-end automated test script verifying multi-subsystem
coordination (Environment, Core Knowledge Matrix, Symbolic Logic Engine)
under Z3 logical enforcement across 100 steps.

Target: Python 3.10
"""

from __future__ import annotations

import os
import random
import sys
import traceback
from typing import Callable, List, Tuple

# Ensure project root is on sys.path when run from tests/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine
from environment import Action, CustomRPGEnv
from environment.entities import TileType
from main import get_forward_tile_context, update_knowledge_from_obs
from ui.web_inspector import WebMindInspector

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_phase3_mind.html")
_TOTAL_STEPS = 100

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
# Shared fixture variables
# ──────────────────────────────────────────────
_env: CustomRPGEnv | None = None
_matrix: CoreKnowledgeMatrix | None = None
_inspector: WebMindInspector | None = None
_sle: SymbolicLogicEngine | None = None

_unsat_blocked_count = 0
_sat_executed_count = 0


def test_initialization() -> None:
    """Test 1 — Instantiate systems and load configurations."""
    global _env, _matrix, _inspector, _sle
    
    _env = CustomRPGEnv()
    
    _matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
    _inspector = WebMindInspector(_matrix)
    
    _sle = SymbolicLogicEngine()
    loaded_count = _sle.load_rules_from_config("config/innate_instincts.json")
    
    assert loaded_count == 4, f"Expected to load 4 rules, got {loaded_count}"
    assert len(_sle.loaded_rules) == 4, "Loaded rules array doesn't match"


def test_execution_loop() -> None:
    """Test 2 — 100 Step Random Execution with Z3 Safety Gate."""
    global _unsat_blocked_count, _sat_executed_count
    
    obs, info = _env.reset(seed=42)
    update_knowledge_from_obs(_matrix, obs, prev_pos=None)
    
    for _ in range(_TOTAL_STEPS):
        action = _env.action_space.sample()
        prev_pos = tuple(obs["player_state"]["position"])
        
        # ── Executive Safety Gate ──
        if action == Action.MOVE_FORWARD:
            state_context = get_forward_tile_context(_env, obs)
            is_safe, explanation, status, active_rules = _sle.verify_action_dynamic(
                "MOVE_FORWARD", state_context
            )
            
            if not is_safe:
                # UNSAT: blocked
                _unsat_blocked_count += 1
                
                # Check invariant: position should not have changed if we didn't call step
                assert tuple(obs["player_state"]["position"]) == prev_pos, "Position changed despite blocked action!"
                continue
                
        # Execute safe action
        obs, reward, terminated, truncated, info = _env.step(action)
        update_knowledge_from_obs(_matrix, obs, prev_pos=prev_pos)
        _sat_executed_count += 1
        
        # Check invariant: never step on WALL or HAZARD
        px, py = obs["player_state"]["position"]
        current_tile_val = int(_env.unwrapped._grid[px, py])
        assert current_tile_val != int(TileType.WALL), f"Safety invariant violation: Stepped on WALL at ({px}, {py})"
        assert current_tile_val != int(TileType.HAZARD), f"Safety invariant violation: Stepped on HAZARD at ({px}, {py})"
        
        if terminated or truncated:
            obs, info = _env.reset()
            update_knowledge_from_obs(_matrix, obs, prev_pos=None)


def test_post_execution_assertions() -> None:
    """Test 3 — Assert safe/unsafe counts and invariants."""
    print(f"       Blocked (UNSAT): {_unsat_blocked_count}")
    print(f"       Executed (SAT):  {_sat_executed_count}")
    
    assert _unsat_blocked_count > 0, "No actions were blocked! Expected Z3 to catch illegal random moves."
    assert _sat_executed_count > 0, "No actions were executed! Expected some valid random moves."
    assert _unsat_blocked_count + _sat_executed_count == _TOTAL_STEPS, "Total step mismatch"


def test_mind_inspector_export() -> None:
    """Test 4 — Export HTML mind map and verify contents."""
    output_path = _inspector.render_html(_OUTPUT_HTML)
    
    assert os.path.isfile(output_path), f"HTML file not found at {output_path}"
    
    size = os.path.getsize(output_path)
    assert size > 0, "HTML file is empty"
    
    # Optional file content checks
    with open(output_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    assert "SPATIAL" in html
    assert "CONNECTS_TO" in html
    assert "pyvis" in html.lower() or "vis-network" in html.lower()


def test_cleanup() -> None:
    """Test 5 — Remove generated HTML artifact."""
    if os.path.isfile(_OUTPUT_HTML):
        os.remove(_OUTPUT_HTML)
    assert not os.path.isfile(_OUTPUT_HTML), f"Failed to clean up {_OUTPUT_HTML}"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  PHASE 3 INTEGRATION VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Initialization & Rules", test_initialization),
        ("Test 2 — 100 Step Z3 Safety Gate Loop", test_execution_loop),
        ("Test 3 — Post-Execution Invariant Assertions", test_post_execution_assertions),
        ("Test 4 — Mind Inspector HTML Export", test_mind_inspector_export),
        ("Test 5 — Cleanup HTML Artifact", test_cleanup),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Phase 3 Integration verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
