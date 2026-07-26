#!/usr/bin/env python3
"""
test_step2_1.py — Core Knowledge Matrix Unit Verification
==========================================================
Zero-dependency standalone test script for Subsystem C
(``CoreKnowledgeMatrix``) and its innate instincts config.

Target: Python 3.10
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Tuple

from cognitive.core_graph import CoreKnowledgeMatrix


# ──────────────────────────────────────────────
# Test runner infrastructure
# ──────────────────────────────────────────────

_results: List[Tuple[str, bool, str]] = []


def _run_test(name: str, fn: Callable[[], None]) -> None:
    """Execute a single test function and record the result."""
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
# Tests
# ──────────────────────────────────────────────

def test_innate_instinct_ingestion() -> None:
    """Test 1 — Verify innate axiom nodes are loaded from JSON."""
    ckm = CoreKnowledgeMatrix("config/innate_instincts.json")

    expected_axioms = [
        "wall_blocking",
        "goal_priority",
        "locked_door_requires_key",
        "hazard_avoidance",
    ]

    for rule_id in expected_axioms:
        assert rule_id in ckm.graph.nodes, (
            f"Axiom node '{rule_id}' missing from graph"
        )
        node = ckm.graph.nodes[rule_id]
        assert node["node_type"] == "AXIOM", (
            f"Node '{rule_id}' has node_type='{node['node_type']}', "
            f"expected 'AXIOM'"
        )
        assert "premise" in node, f"'{rule_id}' missing 'premise' attribute"
        assert "conclusion" in node, f"'{rule_id}' missing 'conclusion' attribute"
        assert "confidence" in node, f"'{rule_id}' missing 'confidence' attribute"
        assert isinstance(node["confidence"], float), (
            f"'{rule_id}' confidence is {type(node['confidence'])}, expected float"
        )

    # Spot-check specific values
    wall = ckm.graph.nodes["wall_blocking"]
    assert wall["premise"] == "InFront(Wall)"
    assert wall["conclusion"] == "Forbid(MOVE_FORWARD)"
    assert wall["confidence"] == 1.0

    hazard = ckm.graph.nodes["hazard_avoidance"]
    assert hazard["confidence"] == 1.0

    goal = ckm.graph.nodes["goal_priority"]
    assert goal["confidence"] == 0.95


def test_spatial_and_entity_node_creation() -> None:
    """Test 2 — Verify spatial and entity node creation with attributes."""
    ckm = CoreKnowledgeMatrix()

    # ── Spatial nodes ──
    n1 = ckm.add_spatial_node(1, 1, "EMPTY")
    n2 = ckm.add_spatial_node(1, 2, "EMPTY")
    n3 = ckm.add_spatial_node(2, 2, "KEY", explored=False)

    assert n1 == "Tile_1_1", f"Expected 'Tile_1_1', got '{n1}'"
    assert n2 == "Tile_1_2", f"Expected 'Tile_1_2', got '{n2}'"
    assert n3 == "Tile_2_2", f"Expected 'Tile_2_2', got '{n3}'"

    node1 = ckm.graph.nodes[n1]
    assert node1["node_type"] == "SPATIAL"
    assert node1["pos"] == (1, 1)
    assert node1["tile_type"] == "EMPTY"
    assert node1["explored"] is True

    node3 = ckm.graph.nodes[n3]
    assert node3["tile_type"] == "KEY"
    assert node3["explored"] is False

    # ── Entity nodes ──
    e1 = ckm.add_entity_node("Key_Red", "KEY", {"color": "red"})
    e2 = ckm.add_entity_node("Door_Red", "DOOR", {"is_locked": True})

    assert e1 == "Key_Red"
    assert e2 == "Door_Red"

    ent1 = ckm.graph.nodes[e1]
    assert ent1["node_type"] == "ENTITY"
    assert ent1["entity_type"] == "KEY"
    assert ent1["color"] == "red"

    ent2 = ckm.graph.nodes[e2]
    assert ent2["node_type"] == "ENTITY"
    assert ent2["entity_type"] == "DOOR"
    assert ent2["is_locked"] is True


def test_typed_directed_edges() -> None:
    """Test 3 — Verify typed directed edges with relation attributes."""
    ckm = CoreKnowledgeMatrix()

    # Build a small topology
    ckm.add_spatial_node(1, 1, "EMPTY")
    ckm.add_spatial_node(1, 2, "EMPTY")
    ckm.add_spatial_node(2, 2, "KEY")
    ckm.add_entity_node("Key_Red", "KEY")
    ckm.add_entity_node("Door_Red", "DOOR")

    # Spatial connectivity
    ckm.add_typed_edge("Tile_1_1", "Tile_1_2", "CONNECTS_TO")
    ckm.add_typed_edge("Tile_1_2", "Tile_2_2", "CONNECTS_TO")

    # Semantic relationships
    ckm.add_typed_edge("Tile_1_2", "Key_Red", "CONTAINS")
    ckm.add_typed_edge("Key_Red", "Door_Red", "OPENS", weight=0.5)

    # Assert CONNECTS_TO edges
    e1 = ckm.graph.edges["Tile_1_1", "Tile_1_2"]
    assert e1["relation"] == "CONNECTS_TO", (
        f"Expected 'CONNECTS_TO', got '{e1['relation']}'"
    )
    assert e1["weight"] == 1.0  # default weight

    e2 = ckm.graph.edges["Tile_1_2", "Tile_2_2"]
    assert e2["relation"] == "CONNECTS_TO"

    # Assert semantic edges
    e3 = ckm.graph.edges["Tile_1_2", "Key_Red"]
    assert e3["relation"] == "CONTAINS"

    e4 = ckm.graph.edges["Key_Red", "Door_Red"]
    assert e4["relation"] == "OPENS"
    assert e4["weight"] == 0.5, (
        f"Custom weight: expected 0.5, got {e4['weight']}"
    )

    # Assert directionality — reverse edges must NOT exist
    assert not ckm.graph.has_edge("Tile_1_2", "Tile_1_1"), (
        "Reverse edge Tile_1_2→Tile_1_1 should not exist"
    )
    assert not ckm.graph.has_edge("Door_Red", "Key_Red"), (
        "Reverse edge Door_Red→Key_Red should not exist"
    )


def test_graph_querying_and_pathfinding() -> None:
    """Test 4 — Verify adjacency queries and shortest-path routing."""
    ckm = CoreKnowledgeMatrix()

    ckm.add_spatial_node(1, 1, "EMPTY")
    ckm.add_spatial_node(1, 2, "EMPTY")
    ckm.add_spatial_node(2, 2, "KEY")

    ckm.add_typed_edge("Tile_1_1", "Tile_1_2", "CONNECTS_TO")
    ckm.add_typed_edge("Tile_1_2", "Tile_2_2", "CONNECTS_TO")

    # ── Adjacent tiles ──
    adj = ckm.get_adjacent_tiles("Tile_1_1")
    assert "Tile_1_2" in adj, (
        f"Expected 'Tile_1_2' in adjacent tiles, got {adj}"
    )

    adj_end = ckm.get_adjacent_tiles("Tile_2_2")
    assert adj_end == [], (
        f"Tile_2_2 has no outgoing CONNECTS_TO, got {adj_end}"
    )

    adj_missing = ckm.get_adjacent_tiles("nonexistent_node")
    assert adj_missing == [], (
        f"Nonexistent node should return [], got {adj_missing}"
    )

    # ── Shortest path ──
    path = ckm.find_shortest_path("Tile_1_1", "Tile_2_2")
    assert path == ["Tile_1_1", "Tile_1_2", "Tile_2_2"], (
        f"Expected 3-hop path, got {path}"
    )

    # No-path case (reverse direction)
    no_path = ckm.find_shortest_path("Tile_2_2", "Tile_1_1")
    assert no_path == [], (
        f"Expected empty list (no reverse edges), got {no_path}"
    )

    # Missing node case
    missing = ckm.find_shortest_path("ghost_a", "ghost_b")
    assert missing == [], (
        f"Expected empty list for missing nodes, got {missing}"
    )


def test_summary_metrics() -> None:
    """Test 5 — Verify graph summary structure and counts."""
    ckm = CoreKnowledgeMatrix()

    # 4 axiom nodes pre-loaded
    s0 = ckm.get_graph_summary()
    assert s0["total_nodes"] == 4, (
        f"Expected 4 axiom nodes at init, got {s0['total_nodes']}"
    )
    assert s0["total_edges"] == 0
    assert s0["node_type_counts"]["AXIOM"] == 4

    # Add spatial + entity nodes
    ckm.add_spatial_node(1, 1, "EMPTY")
    ckm.add_spatial_node(1, 2, "EMPTY")
    ckm.add_spatial_node(2, 2, "KEY")
    ckm.add_entity_node("Key_Red", "KEY")
    ckm.add_entity_node("Door_Red", "DOOR")

    # Add edges
    ckm.add_typed_edge("Tile_1_1", "Tile_1_2", "CONNECTS_TO")
    ckm.add_typed_edge("Tile_1_2", "Tile_2_2", "CONNECTS_TO")
    ckm.add_typed_edge("Tile_1_2", "Key_Red", "CONTAINS")
    ckm.add_typed_edge("Key_Red", "Door_Red", "OPENS")

    s = ckm.get_graph_summary()

    # Structure assertions
    assert "total_nodes" in s, "Missing 'total_nodes' key"
    assert "total_edges" in s, "Missing 'total_edges' key"
    assert "node_type_counts" in s, "Missing 'node_type_counts' key"

    # Count assertions
    assert s["total_nodes"] == 9, (
        f"Expected 9 nodes (4 axiom + 3 spatial + 2 entity), "
        f"got {s['total_nodes']}"
    )
    assert s["total_edges"] == 4, (
        f"Expected 4 edges, got {s['total_edges']}"
    )
    assert s["node_type_counts"]["AXIOM"] == 4
    assert s["node_type_counts"]["SPATIAL"] == 3
    assert s["node_type_counts"]["ENTITY"] == 2


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STEP 2.1 — CORE KNOWLEDGE MATRIX UNIT VERIFICATION")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("Test 1 — Innate instinct ingestion", test_innate_instinct_ingestion),
        ("Test 2 — Spatial & entity node creation", test_spatial_and_entity_node_creation),
        ("Test 3 — Typed directed edges", test_typed_directed_edges),
        ("Test 4 — Graph querying & pathfinding", test_graph_querying_and_pathfinding),
        ("Test 5 — Summary metrics", test_summary_metrics),
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
        print(f"  {passed}/{len(_results)} TESTS PASSED — Core Knowledge Matrix verified.")
    print("══════════════════════════════════════════════════════════════")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
