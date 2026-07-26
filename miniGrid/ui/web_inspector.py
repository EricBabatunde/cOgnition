"""
web_inspector.py — Interactive HTML Graph Mind-Map Inspector
=============================================================
Renders the ``CoreKnowledgeMatrix`` (NetworkX DiGraph) into an
interactive HTML visualiser using PyVis, enabling browser-based
exploration of the agent's spatial, entity, and axiom knowledge.

Target: Python 3.10
"""

from __future__ import annotations

import os
from typing import Any, Dict

from pyvis.network import Network

from cognitive.core_graph import CoreKnowledgeMatrix

# ──────────────────────────────────────────────
# Node styling constants by node_type
# ──────────────────────────────────────────────
_NODE_STYLES: Dict[str, Dict[str, Any]] = {
    "AXIOM": {
        "color": "#E74C3C",
        "shape": "diamond",
        "size": 25,
    },
    "SPATIAL": {
        "color": "#3498DB",
        "shape": "dot",
        "size": 15,
    },
    "ENTITY": {
        "color": "#2ECC71",
        "shape": "box",
        "size": 20,
    },
}

_DEFAULT_STYLE: Dict[str, Any] = {
    "color": "#95A5A6",
    "shape": "ellipse",
    "size": 15,
}

_EDGE_COLOR: str = "#7F8C8D"


class WebMindInspector:
    """Interactive HTML renderer for the Core Knowledge Matrix.

    Converts the agent's internal ``networkx.DiGraph`` into a
    browser-friendly interactive graph using PyVis with physics
    stabilisation for clean node spacing.

    Attributes:
        matrix: Reference to the backing ``CoreKnowledgeMatrix``.
    """

    def __init__(self, matrix: CoreKnowledgeMatrix) -> None:
        """Initialise the inspector with a knowledge matrix.

        Args:
            matrix: The ``CoreKnowledgeMatrix`` instance to render.
        """
        self.matrix: CoreKnowledgeMatrix = matrix

    # ────────────────────────────────────────────
    #  Tooltip builder
    # ────────────────────────────────────────────

    @staticmethod
    def _build_tooltip(node_id: str, attrs: Dict[str, Any]) -> str:
        """Format node attributes into a readable tooltip string.

        Args:
            node_id: The node identifier.
            attrs:   Dictionary of node attributes.

        Returns:
            Multi-line HTML string for the PyVis ``title`` field.
        """
        lines = [f"<b>{node_id}</b>", ""]
        for key, val in attrs.items():
            lines.append(f"<b>{key}:</b> {val}")
        return "<br>".join(lines)

    # ────────────────────────────────────────────
    #  HTML renderer
    # ────────────────────────────────────────────

    def render_html(
        self,
        output_filename: str = "graph_mind.html",
        height: str = "750px",
        width: str = "100%",
    ) -> str:
        """Render the knowledge graph to an interactive HTML file.

        Nodes are colour-coded by type (AXIOM=red, SPATIAL=blue,
        ENTITY=green) and edges carry labelled relation arrows.
        Barnes-Hut physics simulation is enabled for automatic
        layout stabilisation.

        Args:
            output_filename: Filename (or path) for the output HTML.
            height: CSS height string for the visualisation canvas.
            width:  CSS width string for the visualisation canvas.

        Returns:
            Absolute filesystem path to the generated HTML file.
        """
        net = Network(
            directed=True,
            height=height,
            width=width,
            notebook=False,
        )

        # Enable Barnes-Hut physics for clean node spacing
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=120,
            spring_strength=0.05,
            damping=0.09,
            overlap=0,
        )

        graph = self.matrix.graph

        # ── Add nodes ──
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("node_type", "UNKNOWN")
            style = _NODE_STYLES.get(node_type, _DEFAULT_STYLE)

            tooltip = self._build_tooltip(node_id, attrs)

            net.add_node(
                node_id,
                label=node_id,
                title=tooltip,
                color=style["color"],
                shape=style["shape"],
                size=style["size"],
            )

        # ── Add edges ──
        for source, target, edge_data in graph.edges(data=True):
            relation = edge_data.get("relation", "")
            weight = edge_data.get("weight", 1.0)

            net.add_edge(
                source,
                target,
                label=relation,
                arrows="to",
                color=_EDGE_COLOR,
                width=max(1.0, weight),
            )

        # ── Write HTML ──
        net.write_html(output_filename, open_browser=False)

        return os.path.abspath(output_filename)
