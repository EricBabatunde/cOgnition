#!/usr/bin/env python3
"""
test_phase3_stress.py — Phase 3 Z3 Solver Stress & Latency Benchmark
======================================================================
Automated stress and performance benchmark script evaluating the
computational throughput and memory stability of SymbolicLogicEngine
under high-frequency verification calls.

Target: Python 3.10
"""

from __future__ import annotations

import os
import random
import sys
import time
import traceback
from typing import Callable, List, Tuple

import numpy as np

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
# Test cases
# ──────────────────────────────────────────────

def test_z3_stress_benchmark() -> None:
    """Test 1 — 5,000 Z3 Verification Calls Benchmark"""
    sle = SymbolicLogicEngine()
    sle.load_rules_from_config("config/innate_instincts.json")

    iterations = 5000
    latencies_ns: List[int] = []
    
    sat_count = 0
    unsat_count = 0
    accuracy_errors = 0

    contexts = [
        ({"is_wall": True}, False),  # Wall check -> UNSAT
        ({"is_hazard": True}, False),  # Hazard check -> UNSAT
        ({"is_door": True, "is_locked": True, "has_key": False}, False),  # Locked door -> UNSAT
        ({"is_door": False, "is_wall": False, "is_hazard": False}, True),  # Clear path -> SAT
    ]

    t_start_total = time.perf_counter()

    for _ in range(iterations):
        ctx, expected_safe = random.choice(contexts)
        
        t0 = time.perf_counter_ns()
        safe, _, status, _ = sle.verify_action_dynamic("MOVE_FORWARD", ctx)
        t1 = time.perf_counter_ns()
        
        latencies_ns.append(t1 - t0)
        
        if safe != expected_safe:
            accuracy_errors += 1
            
        if status == "SAT":
            sat_count += 1
        elif status == "UNSAT":
            unsat_count += 1

    t_end_total = time.perf_counter()

    # ── Calculate Metrics ──
    total_runtime_s = t_end_total - t_start_total
    throughput = iterations / total_runtime_s
    
    latencies_ms = np.array(latencies_ns) / 1_000_000.0
    mean_latency = np.mean(latencies_ms)
    p95_latency = np.percentile(latencies_ms, 95)
    p99_latency = np.percentile(latencies_ms, 99)
    
    # ── Display Metrics Table ──
    print(f"       Total Iterations: {iterations}")
    print(f"       Total Runtime:    {total_runtime_s:.4f} seconds")
    print(f"       Throughput:       {throughput:.2f} evals/sec")
    print(f"       Mean Latency:     {mean_latency:.2f} ms")
    print(f"       P95 Latency:      {p95_latency:.2f} ms")
    print(f"       P99 Latency:      {p99_latency:.2f} ms")
    print(f"       SAT decisions:    {sat_count}")
    print(f"       UNSAT decisions:  {unsat_count}")
    print(f"       Accuracy Errors:  {accuracy_errors}")
    print()

    # ── Performance Assertions ──
    assert accuracy_errors == 0, f"Accuracy failed: {accuracy_errors} errors out of {iterations}"
    
    assert mean_latency < 2.0, f"Mean latency SLA missed: {mean_latency:.2f}ms >= 2.0ms"
    print("       ✓ Mean Latency SLA (< 2.0 ms) met.")
    
    assert p99_latency < 10.0, f"P99 latency SLA missed: {p99_latency:.2f}ms >= 10.0ms"
    print("       ✓ P99 Latency SLA (< 10.0 ms) met.")
    
    assert throughput > 500, f"Throughput SLA missed: {throughput:.2f} evals/sec <= 500"
    print("       ✓ Throughput SLA (> 500 evals/sec) met.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  PHASE 3 Z3 SOLVER STRESS & LATENCY BENCHMARK")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Z3 Dynamic Verification 5k Calls Benchmark", test_z3_stress_benchmark),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Performance SLAs verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
