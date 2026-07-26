#!/usr/bin/env python3
"""
test_phase2_integration.py — Phase 2 Integration Verification
=============================================================
End-to-end automated test script verifying real-time knowledge
matrix generation and HTML mind map export across 100 environment
steps.

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
from environment import CustomRPGEnv
from main import update_knowledge_from_obs
from ui.web_inspector import WebMindInspector


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_phase2_mind.html")
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


def test_initialization_and_axioms() -> None:
    """Test 1 — Instantiate environment and graph, verify initial axioms."""
    global _env, _matrix, _inspector
    
    _env = CustomRPGEnv()
    _matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
    _inspector = WebMindInspector(_matrix)
    
    summary = _matrix.get_graph_summary()
    assert summary["total_nodes"] == 4, f"Expected 4 axiom nodes, got {summary['total_nodes']}"
    assert summary["node_type_counts"].get("AXIOM") == 4, "Missing AXIOM nodes in counts"
    
    for rule_id in ["wall_blocking", "goal_priority", "locked_door_requires_key", "hazard_avoidance"]:
        assert rule_id in _matrix.graph.nodes, f"Axiom node '{rule_id}' missing"


def test_execution_loop() -> None:
    """Test 2 — Execute 100 steps with dynamic graph grounding."""
    obs, info = _env.reset(seed=42)
    update_knowledge_from_obs(_matrix, obs, prev_pos=None)
    
    for i in range(_TOTAL_STEPS):
        action = _env.action_space.sample()
        prev_pos = tuple(obs["player_state"]["position"])
        
        obs, reward, terminated, truncated, info = _env.step(action)
        update_knowledge_from_obs(_matrix, obs, prev_pos=prev_pos)
        
        if terminated or truncated:
            obs, info = _env.reset()
            update_knowledge_from_obs(_matrix, obs, prev_pos=None)


def test_post_execution_graph_state() -> None:
    """Test 3 — Verify dynamically added nodes and edges."""
    summary = _matrix.get_graph_summary()
    assert summary["total_nodes"] > 4, f"Graph didn't grow, total_nodes={summary['total_nodes']}"
    assert summary["total_edges"] > 0, f"Graph has no edges, total_edges={summary['total_edges']}"
    assert summary["node_type_counts"].get("SPATIAL", 0) > 0, "No SPATIAL nodes created"
    
    # Check at least one spatial node
    spatial_nodes = [n for n, d in _matrix.graph.nodes(data=True) if d.get("node_type") == "SPATIAL"]
    assert len(spatial_nodes) > 0, "Could not find any nodes with node_type == 'SPATIAL'"
    sample_node = spatial_nodes[0]
    assert _matrix.graph.nodes[sample_node]["node_type"] == "SPATIAL"


def test_mind_inspector_export() -> None:
    """Test 4 — Export HTML mind map and verify contents."""
    output_path = _inspector.render_html(_OUTPUT_HTML)
    
    assert os.path.isfile(output_path), f"HTML file not found at {output_path}"
    
    size = os.path.getsize(output_path)
    assert size > 0, "HTML file is empty"
    
    with open(output_path, "r", encoding="utf-8") as fh:
        html_content = fh.read()
        
    assert "SPATIAL" in html_content, "Missing 'SPATIAL' text in HTML"
    assert "CONNECTS_TO" in html_content, "Missing 'CONNECTS_TO' edge label in HTML"
    
    has_js = "pyvis" in html_content.lower() or "vis-network" in html_content.lower() or "vis.min" in html_content.lower()
    assert has_js, "Missing pyvis / vis-network JS dependencies"


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
    print("  PHASE 2 INTEGRATION VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Initialization & Axioms", test_initialization_and_axioms),
        ("Test 2 — 100 Step Execution Loop", test_execution_loop),
        ("Test 3 — Post-Execution Graph State", test_post_execution_graph_state),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Phase 2 Integration verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
