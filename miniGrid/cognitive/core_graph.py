"""
core_graph.py — Core Knowledge Matrix (Subsystem C)
=====================================================
Implements a persistent directed graph representing the agent's
spatial and causal world model using NetworkX.

The graph stores three node types:
  - **AXIOM**: Innate instinct rules loaded from JSON config.
  - **SPATIAL**: Tile positions on the 10×10 grid.
  - **ENTITY**: In-world objects (keys, doors, hazards, goals).

Edges encode typed relations: ``CONNECTS_TO``, ``CONTAINS``,
``REQUIRES``, ``OPENS``, ``CAUSES_DAMAGE``, etc.

Target: Python 3.10
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


class CoreKnowledgeMatrix:
    """Persistent directed knowledge graph for the cognitive engine.

    Manages spatial topology, entity relationships, and innate
    axiom rules within a single ``networkx.DiGraph``.

    Attributes:
        graph: The directed graph backing all knowledge storage.
    """

    def __init__(
        self,
        config_path: str = "config/innate_instincts.json",
    ) -> None:
        """Initialise an empty knowledge graph and load innate axioms.

        Args:
            config_path: Path to the innate instincts JSON file.
                         Resolved relative to the project root.
        """
        self.graph: nx.DiGraph = nx.DiGraph()
        self._load_innate_instincts(config_path)

    # ────────────────────────────────────────────
    #  Innate instinct loader
    # ────────────────────────────────────────────

    def _load_innate_instincts(self, config_path: str) -> None:
        """Parse the innate instincts JSON and seed AXIOM nodes.

        Each rule becomes a graph node with ``node_type="AXIOM"``
        and attributes for premise, conclusion, and confidence.

        Args:
            config_path: Filesystem path to the JSON config.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        # Resolve relative paths against the directory containing
        # this source file so it works regardless of cwd.
        if not os.path.isabs(config_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, config_path)

        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for rule in data.get("innate_rules", []):
            rule_id: str = rule["rule_id"]
            self.graph.add_node(
                rule_id,
                node_type="AXIOM",
                premise=rule["premise"],
                conclusion=rule["conclusion"],
                confidence=rule["confidence"],
            )

    # ────────────────────────────────────────────
    #  Spatial node management
    # ────────────────────────────────────────────

    def add_spatial_node(
        self,
        x: int,
        y: int,
        tile_type: str,
        explored: bool = True,
    ) -> str:
        """Add or update a spatial tile node in the graph.

        Args:
            x:         Column coordinate.
            y:         Row coordinate.
            tile_type: Tile classification string (e.g. ``"WALL"``,
                       ``"EMPTY"``, ``"DOOR"``).
            explored:  Whether the tile has been observed by the agent.

        Returns:
            The unique node identifier ``Tile_{x}_{y}``.
        """
        node_id = f"Tile_{x}_{y}"
        self.graph.add_node(
            node_id,
            node_type="SPATIAL",
            pos=(x, y),
            tile_type=tile_type,
            explored=explored,
        )
        return node_id

    # ────────────────────────────────────────────
    #  Entity node management
    # ────────────────────────────────────────────

    def add_entity_node(
        self,
        entity_id: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add or update an entity node in the graph.

        Args:
            entity_id:   Unique entity identifier (e.g. ``"key_red"``).
            entity_type: Classification (e.g. ``"KEY"``, ``"DOOR"``).
            attributes:  Optional extra attributes to store on the node.

        Returns:
            The ``entity_id`` used as the node key.
        """
        attrs: Dict[str, Any] = {
            "node_type": "ENTITY",
            "entity_type": entity_type,
        }
        if attributes:
            attrs.update(attributes)

        self.graph.add_node(entity_id, **attrs)
        return entity_id

    # ────────────────────────────────────────────
    #  Edge management
    # ────────────────────────────────────────────

    def add_typed_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
    ) -> None:
        """Add a directed, typed edge between two nodes.

        If the edge already exists it is updated in-place.

        Args:
            source_id: Source node identifier.
            target_id: Target node identifier.
            relation:  Semantic relation label (e.g. ``"CONNECTS_TO"``,
                       ``"CONTAINS"``, ``"REQUIRES"``, ``"OPENS"``,
                       ``"CAUSES_DAMAGE"``).
            weight:    Optional numeric weight for path-finding.
        """
        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation,
            weight=weight,
        )

    # ────────────────────────────────────────────
    #  Queries
    # ────────────────────────────────────────────

    def get_adjacent_tiles(self, node_id: str) -> List[str]:
        """Return spatial neighbours connected by CONNECTS_TO edges.

        Args:
            node_id: Source node to query.

        Returns:
            List of adjacent spatial node IDs. Empty list if none
            exist or the node is not in the graph.
        """
        if node_id not in self.graph:
            return []

        result: List[str] = []
        for _, target, data in self.graph.edges(node_id, data=True):
            if data.get("relation") == "CONNECTS_TO":
                target_data = self.graph.nodes.get(target, {})
                if target_data.get("node_type") == "SPATIAL":
                    result.append(target)
        return result

    def find_shortest_path(
        self,
        start_node: str,
        target_node: str,
    ) -> List[str]:
        """Compute the shortest topological path between two nodes.

        Uses ``networkx.shortest_path`` with uniform edge weights.

        Args:
            start_node:  Source node ID.
            target_node: Destination node ID.

        Returns:
            Ordered list of node IDs from start to target
            (inclusive). Empty list if no path exists or either
            node is missing.
        """
        try:
            return nx.shortest_path(self.graph, start_node, target_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ────────────────────────────────────────────
    #  Diagnostics
    # ────────────────────────────────────────────

    def get_graph_summary(self) -> Dict[str, Any]:
        """Return a diagnostic summary of the knowledge graph.

        Returns:
            Dictionary with keys:
              - ``total_nodes``: int
              - ``total_edges``: int
              - ``node_type_counts``: Dict mapping node_type → count
        """
        type_counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            ntype = data.get("node_type", "UNKNOWN")
            type_counts[ntype] = type_counts.get(ntype, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_type_counts": type_counts,
        }
