#!/usr/bin/env python3
"""
test_step4_1.py — Fast Plasticity Memory Unit Verification
===========================================================
Automated unit test script to verify FAISS vector indexing,
sub-millisecond retrieval latency, ring buffer capacity limits,
and novelty score calculations in FastPlasticityMemory.

Target: Python 3.10
"""

from __future__ import annotations

import os
import sys
import time
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

def test_vectorization_and_storage() -> None:
    """Test 1 — Vectorization & Storage (10 synthetic experiences)."""
    mem = FastPlasticityMemory(dimension=64, capacity=100)
    
    for i in range(10):
        obs = {
            "player_state": {
                "position": (i, i),
                "direction": 1,
                "health": 100,
                "inventory": []
            }
        }
        ctx = {
            "is_wall": False,
            "is_hazard": False,
            "is_door": False,
            "is_locked": False,
            "has_key": False,
            "target_tile": "EMPTY"
        }
        mem.store_experience(obs, ctx, "MOVE_FORWARD", 0.0)
        
    assert len(mem.experience_buffer) == 10, (
        f"Expected 10 in buffer, got {len(mem.experience_buffer)}"
    )
    assert mem.faiss_index.ntotal == 10, (
        f"Expected 10 in FAISS, got {mem.faiss_index.ntotal}"
    )

    print(f"       Stored: 10 experiences")
    print(f"       Buffer Size: {len(mem.experience_buffer)}")
    print(f"       FAISS Count: {mem.faiss_index.ntotal}")


def test_latency_retrieval() -> None:
    """Test 2 — Sub-Millisecond Nearest Neighbor Retrieval."""
    mem = FastPlasticityMemory(dimension=64, capacity=100)
    
    # Store some context
    obs = {
        "player_state": {
            "position": (5, 5),
            "direction": 0,
            "health": 100,
            "inventory": ["key_red"]
        }
    }
    ctx = {
        "is_wall": False,
        "is_hazard": False,
        "is_door": True,
        "is_locked": True,
        "has_key": True,
        "target_tile": "DOOR"
    }
    mem.store_experience(obs, ctx, "MOVE_FORWARD", 0.0)
    
    # Vectorize same query
    query_vec = mem.vectorizer.vectorize(obs, ctx)
    
    # Warmup FAISS to avoid OpenBLAS/OpenMP thread pool initialization latency
    _ = mem.query_similar(query_vec, k=1)
    
    # Measure latency
    t0 = time.perf_counter_ns()
    matches = mem.query_similar(query_vec, k=1)
    t1 = time.perf_counter_ns()
    
    latency_ms = (t1 - t0) / 1_000_000.0
    
    assert latency_ms < 1.0, f"Retrieval latency too high: {latency_ms:.3f} ms >= 1.0 ms"
    assert len(matches) > 0, "No matches returned"
    
    top_exp, top_dist = matches[0]
    assert top_dist < 0.25, f"Expected top distance < 0.25, got {top_dist:.4f}"
    
    print(f"       Latency: {latency_ms:.3f} ms")
    print(f"       Top Match L2 Distance: {top_dist:.4f}")


def test_novelty_score() -> None:
    """Test 3 — Novelty Detection Score ΔE."""
    mem = FastPlasticityMemory(dimension=64, capacity=100)
    
    base_obs = {
        "player_state": {
            "position": (2, 2),
            "direction": 2,
            "health": 100,
            "inventory": []
        }
    }
    base_ctx = {
        "is_wall": False,
        "is_hazard": False,
        "is_door": False,
        "is_locked": False,
        "has_key": False,
        "target_tile": "EMPTY"
    }
    
    # Store identical state a few times to build neighborhood
    for _ in range(5):
        mem.store_experience(base_obs, base_ctx, "MOVE_FORWARD", 0.0)
        
    # Same state novelty
    same_vec = mem.vectorizer.vectorize(base_obs, base_ctx)
    novelty_same = mem.calculate_novelty(same_vec, k=3)
    assert novelty_same < 0.15, f"Expected low novelty for same state, got {novelty_same:.4f}"
    
    # Out of distribution state
    ood_obs = {
        "player_state": {
            "position": (999, -999),  # Extremely different
            "direction": 3,
            "health": 10,
            "inventory": ["a", "b", "c", "d", "e"]
        }
    }
    ood_ctx = {
        "is_wall": True,
        "is_hazard": True,
        "is_door": True,
        "is_locked": True,
        "has_key": True,
        "target_tile": "HAZARD"
    }
    
    ood_vec = mem.vectorizer.vectorize(ood_obs, ood_ctx)
    novelty_ood = mem.calculate_novelty(ood_vec, k=3)
    
    # FAISS Cosine distance (L2 normalized) maxes at 4.0 for completely opposite vectors
    # We expect a significantly higher distance than 0.15
    assert novelty_ood > 0.60, f"Expected high novelty for OOD state, got {novelty_ood:.4f}"
    
    print(f"       Identical state ΔE: {novelty_same:.4f}")
    print(f"       OOD state ΔE: {novelty_ood:.4f}")


def test_ring_buffer_eviction() -> None:
    """Test 4 — Ring Buffer Eviction (Store 150 items, cap at 100)."""
    capacity = 100
    mem = FastPlasticityMemory(dimension=64, capacity=capacity)
    
    for i in range(150):
        obs = {
            "player_state": {
                "position": (i % 10, i % 10),
                "direction": 1,
                "health": 100,
                "inventory": []
            }
        }
        ctx = {
            "is_wall": False,
            "is_hazard": False,
            "is_door": False,
            "is_locked": False,
            "has_key": False,
            "target_tile": "EMPTY"
        }
        mem.store_experience(obs, ctx, "MOVE_FORWARD", 0.0)
        
    assert len(mem.experience_buffer) == capacity, (
        f"Expected {capacity} in buffer, got {len(mem.experience_buffer)}"
    )
    assert mem.faiss_index.ntotal == capacity, (
        f"Expected {capacity} in FAISS, got {mem.faiss_index.ntotal}"
    )

    print(f"       Inserted: 150 items")
    print(f"       Buffer Capacity Limit: {capacity}")
    print(f"       Actual Buffer Size: {len(mem.experience_buffer)}")
    print(f"       Actual FAISS Count: {mem.faiss_index.ntotal}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STEP 4.1 — FAST PLASTICITY MEMORY UNIT VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Vectorization & Storage", test_vectorization_and_storage),
        ("Test 2 — Sub-Millisecond NN Retrieval Latency", test_latency_retrieval),
        ("Test 3 — Novelty Detection Score ΔE", test_novelty_score),
        ("Test 4 — Ring Buffer Eviction", test_ring_buffer_eviction),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Memory Layer verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
