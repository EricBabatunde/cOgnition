#!/usr/bin/env python3
"""
test_step4_2.py — Hebbian Plasticity & Decay Unit Verification
==============================================================
Automated unit test script to verify Hebbian weight reinforcement,
step-wise recency decay, and decay-adjusted retrieval ranking in
FastPlasticityMemory.

Target: Python 3.10
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Callable, List, Tuple

import numpy as np

# Ensure project root is on sys.path when run from tests/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cognitive.fast_memory import FastPlasticityMemory


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
# Test cases
# ──────────────────────────────────────────────

def test_hebbian_reinforcement() -> None:
    """Test 1 — Hebbian Reinforcement Verification."""
    mem = FastPlasticityMemory(dimension=64, capacity=100)
    
    # Reinforce state-action pair ("DOOR", "TOGGLE_INTERACT") 5 times with reward=1.0
    for _ in range(5):
        mem.reinforce_hebbian("DOOR", "TOGGLE_INTERACT", 1.0)
        
    final_weight = mem.hebbian_matrix.get(("DOOR", "TOGGLE_INTERACT"), 0.0)
    
    # Each reinforcement adds `hebbian_learning_rate * (1.0 + reward)`
    # 0.1 * (1.0 + 1.0) = 0.2 per reinforcement. 5 * 0.2 = 1.0.
    assert final_weight > 0.8, f"Expected weight > 0.8, got {final_weight}"
    print(f"       Final Hebbian Weight: {final_weight:.2f}")


def test_recency_decay() -> None:
    """Test 2 — Exponential Recency Decay."""
    mem = FastPlasticityMemory(dimension=64, capacity=100)
    
    obs = {
        "player_state": {
            "position": (1, 1),
            "direction": 0,
            "health": 100,
            "inventory": []
        }
    }
    ctx = {"target_tile": "EMPTY"}
    
    # Store experience E1 at step 0
    exp1 = mem.store_experience(obs, ctx, "MOVE_FORWARD", 0.0)
    
    # Advance clock by 100 steps
    for _ in range(100):
        mem.step_clock()
        
    # Expected decay is 0.995^100 ≈ 0.605
    assert exp1.weight < 0.65, f"Expected E1 weight < 0.65, got {exp1.weight:.3f}"
    print(f"       E1 Weight after 100 steps: {exp1.weight:.3f}")


def test_decay_adjusted_ranking() -> None:
    """Test 3 — Decay-Adjusted Retrieval Ranking."""
    mem = FastPlasticityMemory(dimension=64, capacity=100)
    
    # Query parameters
    base_obs = {"player_state": {"position": (1, 1), "direction": 0, "health": 50, "inventory": []}}
    ctx = {"target_tile": "EMPTY"}
    
    query_vec = mem.vectorizer.vectorize(base_obs, ctx)
    
    # E_old has health 55, close to query (health 50)
    old_obs = {"player_state": {"position": (1, 1), "direction": 0, "health": 55, "inventory": []}}
    e_old = mem.store_experience(old_obs, ctx, "MOVE_FORWARD", 0.0)
    
    # Advance clock by 200 steps
    for _ in range(200):
        mem.step_clock()
        
    # E_new has health 58, slightly further from query (health 50) than E_old
    new_obs = {"player_state": {"position": (1, 1), "direction": 0, "health": 58, "inventory": []}}
    e_new = mem.store_experience(new_obs, ctx, "MOVE_FORWARD", 0.0)
    
    # Raw distances: E_old ~0.0005, E_new ~0.0013
    # With E_old weight ≈ 0.367 and E_new weight = 1.0, 
    # Adjusted distances: E_old ~0.0014, E_new ~0.0013
    # Therefore, E_new should rank ABOVE E_old!
    
    matches = mem.query_similar(query_vec, k=2)
    assert len(matches) == 2, "Expected 2 matches"
    
    top_match = matches[0][0]
    assert top_match.experience_id == e_new.experience_id, "E_new should rank above E_old due to decay!"
    
    print(f"       Top match is E_new: Adjusted Dist = {matches[0][1]:.5f}")
    print(f"       Second match is E_old: Adjusted Dist = {matches[1][1]:.5f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STEP 4.2 — HEBBIAN PLASTICITY & DECAY UNIT VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Hebbian Reinforcement Verification", test_hebbian_reinforcement),
        ("Test 2 — Exponential Recency Decay", test_recency_decay),
        ("Test 3 — Decay-Adjusted Retrieval Ranking", test_decay_adjusted_ranking),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Hebbian mechanics verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
