#!/usr/bin/env python3
"""
test_phase2_graphtest.py — Phase 2 Graph Scale & Latency Stress Test
=====================================================================
Automated performance and scale stress test script evaluating
``CoreKnowledgeMatrix`` and ``WebMindInspector`` under heavy workload
(1,000+ nodes, 3,000+ edges).

Target: Python 3.10
"""

from __future__ import annotations

import os
import random
import sys
import time
import traceback
from typing import Callable, List, Tuple

# Ensure project root is on sys.path when run from tests/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cognitive.core_graph import CoreKnowledgeMatrix
from ui.web_inspector import WebMindInspector

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_graph_mind.html")
_GRID_SIZE = 32
_NUM_ENTITIES = 100
_NUM_PATH_QUERIES = 100
_MAX_LATENCY_MS = 10.0

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
_matrix: CoreKnowledgeMatrix | None = None
_inspector: WebMindInspector | None = None
_spatial_nodes: List[str] = []


def test_scaled_graph_generation() -> None:
    """Test 1 — Scaled Graph Generation (1,024 Spatial + 100 Entity Nodes)."""
    global _matrix, _inspector, _spatial_nodes
    
    _matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
    _inspector = WebMindInspector(_matrix)
    
    # Generate 32x32 spatial grid (1,024 nodes)
    for r in range(_GRID_SIZE):
        for c in range(_GRID_SIZE):
            node_id = _matrix.add_spatial_node(r, c, "EMPTY", explored=True)
            _spatial_nodes.append(node_id)
            
    # Build bidirectional CONNECTS_TO edges
    for r in range(_GRID_SIZE):
        for c in range(_GRID_SIZE):
            src_id = f"Tile_{r}_{c}"
            if c + 1 < _GRID_SIZE:
                dst_id = f"Tile_{r}_{c+1}"
                _matrix.add_typed_edge(src_id, dst_id, "CONNECTS_TO")
                _matrix.add_typed_edge(dst_id, src_id, "CONNECTS_TO")
            if r + 1 < _GRID_SIZE:
                dst_id = f"Tile_{r+1}_{c}"
                _matrix.add_typed_edge(src_id, dst_id, "CONNECTS_TO")
                _matrix.add_typed_edge(dst_id, src_id, "CONNECTS_TO")
                
    # Inject 100 ENTITY nodes
    entity_types = ["KEY", "DOOR", "HAZARD"]
    for i in range(_NUM_ENTITIES):
        ent_type = random.choice(entity_types)
        ent_id = f"Entity_{i}_{ent_type}"
        _matrix.add_entity_node(ent_id, ent_type)
        
        # Link to random spatial node
        host_tile = random.choice(_spatial_nodes)
        _matrix.add_typed_edge(host_tile, ent_id, "CONTAINS")


def test_pathfinding_latency() -> None:
    """Test 2 — Pathfinding Latency Benchmarking."""
    latencies = []
    
    # Divide the grid to pick start/target nodes across opposite ends
    # or just randomly pick from the list.
    for _ in range(_NUM_PATH_QUERIES):
        start = random.choice(_spatial_nodes)
        target = random.choice(_spatial_nodes)
        
        t0 = time.perf_counter()
        path = _matrix.find_shortest_path(start, target)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000)  # ms
        
        assert len(path) > 0 or start == target, f"Failed to find path between {start} and {target}"
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"       Total Queries: {_NUM_PATH_QUERIES}")
    print(f"       Average Pathfinding Latency: {avg_latency:.2f} ms")
    
    assert avg_latency < _MAX_LATENCY_MS, f"Average latency {avg_latency:.2f}ms exceeds maximum {_MAX_LATENCY_MS}ms"


def test_graph_summary_assertion() -> None:
    """Test 3 — Graph Summary Assertion (nodes >= 1,128, edges > 3,000)."""
    summary = _matrix.get_graph_summary()
    
    print(f"       Total Nodes: {summary['total_nodes']}")
    print(f"       Total Edges: {summary['total_edges']}")
    
    assert summary["total_nodes"] >= 1128, f"Expected >= 1128 nodes, got {summary['total_nodes']}"
    assert summary["total_edges"] > 3000, f"Expected > 3000 edges, got {summary['total_edges']}"


def test_heavy_exporter_benchmark() -> None:
    """Test 4 — Heavy Exporter Rendering Benchmark (> 50 KB HTML)."""
    output_path = _inspector.render_html(_OUTPUT_HTML)
    
    assert os.path.isfile(output_path), f"HTML file not found at {output_path}"
    
    size = os.path.getsize(output_path)
    print(f"       Exported File Size: {size / 1024:.2f} KB")
    
    assert size > 50 * 1024, f"HTML file is too small ({size / 1024:.2f} KB), expected > 50 KB"


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
    print("  PHASE 2 GRAPH SCALE & LATENCY STRESS TEST")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Scaled Graph Generation", test_scaled_graph_generation),
        ("Test 2 — Pathfinding Latency Benchmarking", test_pathfinding_latency),
        ("Test 3 — Graph Summary Assertion", test_graph_summary_assertion),
        ("Test 4 — Heavy Exporter Rendering Benchmark", test_heavy_exporter_benchmark),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Graph Scale & Latency verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
