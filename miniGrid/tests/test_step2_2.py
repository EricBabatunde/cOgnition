#!/usr/bin/env python3
"""
test_step2_2.py — WebMindInspector HTML Rendering Verification
================================================================
Verifies that ``WebMindInspector`` correctly renders a populated
``CoreKnowledgeMatrix`` into a valid interactive HTML file with
expected node identifiers, edge labels, and JS dependencies.

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
from ui.web_inspector import WebMindInspector

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_graph_mind.html")

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
_ckm: CoreKnowledgeMatrix | None = None
_html_path: str = ""
_html_content: str = ""


def _setup() -> None:
    """Build a populated graph, render HTML, and cache outputs."""
    global _ckm, _html_path, _html_content

    _ckm = CoreKnowledgeMatrix("config/innate_instincts.json")

    # Spatial nodes
    _ckm.add_spatial_node(1, 1, "EMPTY")
    _ckm.add_spatial_node(1, 2, "EMPTY")

    # Entity node
    _ckm.add_entity_node("Key_Red", "KEY", {"color": "red"})

    # Typed edges
    _ckm.add_typed_edge("Tile_1_1", "Tile_1_2", "CONNECTS_TO")
    _ckm.add_typed_edge("Tile_1_2", "Key_Red", "CONTAINS")

    # Render
    inspector = WebMindInspector(_ckm)
    _html_path = inspector.render_html(_OUTPUT_HTML)

    with open(_html_path, "r", encoding="utf-8") as fh:
        _html_content = fh.read()


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_html_file_exists() -> None:
    """Test 1 — HTML file exists on disk with non-zero size."""
    assert os.path.isfile(_html_path), (
        f"HTML file not found at: {_html_path}"
    )
    size = os.path.getsize(_html_path)
    assert size > 0, f"HTML file is empty (0 bytes)"
    print(f"       File: {_html_path}")
    print(f"       Size: {size:,} bytes")


def test_spatial_nodes_in_html() -> None:
    """Test 2 — Spatial node identifiers appear in the HTML."""
    assert "Tile_1_1" in _html_content, (
        "'Tile_1_1' not found in rendered HTML"
    )
    assert "Tile_1_2" in _html_content, (
        "'Tile_1_2' not found in rendered HTML"
    )


def test_entity_node_in_html() -> None:
    """Test 3 — Entity node identifier appears in the HTML."""
    assert "Key_Red" in _html_content, (
        "'Key_Red' not found in rendered HTML"
    )


def test_axiom_nodes_in_html() -> None:
    """Test 4 — Pre-seeded axiom nodes appear in the HTML."""
    assert "wall_blocking" in _html_content, (
        "'wall_blocking' axiom not found in rendered HTML"
    )
    assert "hazard_avoidance" in _html_content, (
        "'hazard_avoidance' axiom not found in rendered HTML"
    )


def test_edge_labels_in_html() -> None:
    """Test 5 — Typed edge relation labels appear in the HTML."""
    assert "CONNECTS_TO" in _html_content, (
        "'CONNECTS_TO' edge label not found in rendered HTML"
    )
    assert "CONTAINS" in _html_content, (
        "'CONTAINS' edge label not found in rendered HTML"
    )


def test_node_colour_codes_in_html() -> None:
    """Test 6 — Node colour codes for each type are present."""
    assert "#E74C3C" in _html_content, (
        "AXIOM colour #E74C3C not found in HTML"
    )
    assert "#3498DB" in _html_content, (
        "SPATIAL colour #3498DB not found in HTML"
    )
    assert "#2ECC71" in _html_content, (
        "ENTITY colour #2ECC71 not found in HTML"
    )


def test_js_dependencies_in_html() -> None:
    """Test 7 — PyVis / vis-network JS dependencies are embedded."""
    has_pyvis = "pyvis" in _html_content.lower()
    has_vis = "vis-network" in _html_content.lower() or "vis.min" in _html_content.lower()
    has_vis_data = "new vis.Network" in _html_content or "new vis.DataSet" in _html_content
    assert has_pyvis or has_vis or has_vis_data, (
        "No PyVis / vis-network JavaScript references found in HTML"
    )


def test_cleanup() -> None:
    """Test 8 — Clean up generated test HTML file."""
    if os.path.isfile(_OUTPUT_HTML):
        os.remove(_OUTPUT_HTML)
    assert not os.path.isfile(_OUTPUT_HTML), (
        f"Failed to remove test file: {_OUTPUT_HTML}"
    )
    print(f"       Cleaned up: {_OUTPUT_HTML}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STEP 2.2 — WEB MIND INSPECTOR UNIT VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    # Build shared fixture
    _setup()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — HTML file exists on disk", test_html_file_exists),
        ("Test 2 — Spatial nodes in HTML", test_spatial_nodes_in_html),
        ("Test 3 — Entity node in HTML", test_entity_node_in_html),
        ("Test 4 — Axiom nodes in HTML", test_axiom_nodes_in_html),
        ("Test 5 — Edge labels in HTML", test_edge_labels_in_html),
        ("Test 6 — Node colour codes in HTML", test_node_colour_codes_in_html),
        ("Test 7 — JS dependencies in HTML", test_js_dependencies_in_html),
        ("Test 8 — Cleanup test artifact", test_cleanup),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — WebMindInspector verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
