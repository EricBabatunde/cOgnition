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
    #  Topological Pathfinding
    # ────────────────────────────────────────────

    # Clockwise direction ring: NORTH → EAST → SOUTH → WEST
    _DIR_RING: List[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def find_topological_path(
        self,
        start_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        known_inventory: Optional[List[str]] = None,
        exclude_nodes: Optional[set[Tuple[int, int]]] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest traversable path using A* on the knowledge graph.

        Constructs a filtered subgraph view that excludes WALL and
        HAZARD tiles, and only includes DOOR tiles when the agent
        holds a matching KEY in ``known_inventory``.

        Args:
            start_pos:       ``(row, col)`` agent start coordinate.
            target_pos:      ``(row, col)`` desired destination.
            known_inventory: List of inventory item strings. Doors are
                             traversable when any item contains ``"key"``
                             (case-insensitive).

        Returns:
            Ordered list of ``(row, col)`` coordinate tuples from start
            to target (inclusive), or ``None`` if no valid path exists.
        """
        if known_inventory is None:
            known_inventory = []
            
        if exclude_nodes is None:
            exclude_nodes = set()

        has_key = any("key" in item.lower() for item in known_inventory)

        def _is_traversable(node: str) -> bool:
            data = self.graph.nodes.get(node, {})
            if data.get("node_type") != "SPATIAL":
                return False
                
            pos = data.get("pos")
            if pos and tuple(pos) in exclude_nodes:
                return False
                
            tile = data.get("tile_type", "UNKNOWN")
            if tile == "WALL":
                return False
            if tile == "HAZARD":
                return False
            if tile == "DOOR" and not has_key:
                return False
            return True

        # Build subgraph view containing only traversable spatial nodes
        traversable_nodes = [n for n in self.graph.nodes if _is_traversable(n)]
        subgraph: nx.DiGraph = self.graph.subgraph(traversable_nodes)

        start_id = f"Tile_{start_pos[0]}_{start_pos[1]}"
        target_id = f"Tile_{target_pos[0]}_{target_pos[1]}"

        if start_id not in subgraph or target_id not in subgraph:
            return None

        def _l1_heuristic(u: str, v: str) -> float:
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            ux, uy = u_data.get("pos", (0, 0))
            vx, vy = v_data.get("pos", (0, 0))
            return abs(ux - vx) + abs(uy - vy)

        try:
            node_path = nx.astar_path(
                subgraph, start_id, target_id, heuristic=_l1_heuristic
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        # Convert node IDs back to coordinate tuples
        coord_path: List[Tuple[int, int]] = []
        for node_id in node_path:
            pos = self.graph.nodes[node_id].get("pos")
            if pos is not None:
                coord_path.append(pos)
            else:
                # Fallback: parse from node ID "Tile_R_C"
                parts = node_id.split("_")
                coord_path.append((int(parts[1]), int(parts[2])))

        return coord_path

    # ────────────────────────────────────────────
    #  Action Sequence Planner
    # ────────────────────────────────────────────

    def plan_action_sequence(
        self,
        path: List[Tuple[int, int]],
        current_direction: Tuple[int, int],
    ) -> List[str]:
        """Translate a coordinate path into a sequence of action strings.

        Given a list of waypoint coordinates and the agent's current
        facing direction vector ``(dx, dy)``, produces the minimal
        sequence of ``TURN_LEFT``, ``TURN_RIGHT``, and ``MOVE_FORWARD``
        actions required to traverse the path.

        Direction ring (clockwise index):
            0 = NORTH (-1, 0)
            1 = EAST  ( 0, 1)
            2 = SOUTH ( 1, 0)
            3 = WEST  ( 0,-1)

        Args:
            path:              Ordered ``(row, col)`` waypoints
                               (at least 2 entries).
            current_direction: Agent's initial facing as ``(dx, dy)``.

        Returns:
            List of action name strings. Empty list if the path has
            fewer than 2 waypoints.
        """
        if len(path) < 2:
            return []

        ring = self._DIR_RING
        try:
            facing_idx = ring.index(tuple(current_direction))
        except ValueError:
            # Default to NORTH if direction is unrecognised
            facing_idx = 0

        actions: List[str] = []

        for i in range(len(path) - 1):
            curr_r, curr_c = path[i]
            next_r, next_c = path[i + 1]

            target_dr = next_r - curr_r
            target_dc = next_c - curr_c

            try:
                target_idx = ring.index((target_dr, target_dc))
            except ValueError:
                # Skip non-cardinal movements
                continue

            # Calculate minimal rotation in the clockwise ring
            delta = (target_idx - facing_idx) % 4

            if delta == 0:
                # Already facing the target
                pass
            elif delta == 1:
                actions.append("TURN_RIGHT")
            elif delta == 3:
                actions.append("TURN_LEFT")
            elif delta == 2:
                actions.append("TURN_RIGHT")
                actions.append("TURN_RIGHT")

            actions.append("MOVE_FORWARD")
            facing_idx = target_idx

        return actions

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

    # ────────────────────────────────────────────
    #  Serialization
    # ────────────────────────────────────────────

    def save_graph(self, filepath: str = "config/graph_memory.json") -> None:
        """Export current graph nodes, edges, attributes, and synthesized rules."""
        import os
        from networkx.readwrite import json_graph
        
        if not os.path.isabs(filepath):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, filepath)
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = json_graph.node_link_data(self.graph, edges="edges")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_graph(self, filepath: str = "config/graph_memory.json") -> None:
        """Load and reconstruct graph nodes and synthesized rules from disk."""
        import os
        from networkx.readwrite import json_graph
        
        if not os.path.isabs(filepath):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, filepath)
            
        if not os.path.exists(filepath):
            return
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        edge_kwarg = "edges" if "edges" in data else "links"
        self.graph = json_graph.node_link_graph(data, edges=edge_kwarg)
