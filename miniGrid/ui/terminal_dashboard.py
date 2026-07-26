"""
terminal_dashboard.py — Rich Split-Panel Terminal Dashboard
============================================================
Renders a compact 3-section terminal UI for the Neuro-Symbolic 
RPG Cognitive Engine using the ``rich`` library.

Designed to fit within a strict 18-line vertical budget to
prevent terminal overflow or scrolling.

Layout (18 lines total)
-----------------------
::

    ┌──────────────────┬──────────────────────┐  ─┐
    │   WORLD VIEW     │  MEMORY & LOGIC      │   │
    │  (10×10 grid)    │  - Goal / FAISS      │   │ 12 lines
    │                  │  - Rules / Blocked   │   │ (top)
    └──────────────────┴──────────────────────┘  ─┘
    ┌─────────────────────────────────────────┐  ─┐
    │  THOUGHT STREAM (3 logs)                │   │ 5 lines
    └─────────────────────────────────────────┘  ─┘
       HP │ Step │ Inventory │ State             ─┤ 1 line (status)

Target: Python 3.10
"""

from __future__ import annotations

import collections
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from environment.entities import Direction, TileType

# ──────────────────────────────────────────────
# Layout height budget (must total ≤ 18)
# ──────────────────────────────────────────────
_TOP_HEIGHT: int = 12        # World and Memory panels
_THOUGHT_HEIGHT: int = 5     # Thought stream log panel
_STATUS_HEIGHT: int = 1      # Status bar (no borders)
_MAX_HEIGHT: int = _TOP_HEIGHT + _THOUGHT_HEIGHT + _STATUS_HEIGHT

_THOUGHT_LOG_LINES: int = 3  # max visible log entries

# ──────────────────────────────────────────────
# Tile rendering constants (1 char per tile + single space)
# ──────────────────────────────────────────────
_TILE_ICON: Dict[int, str] = {
    TileType.EMPTY:  "· ",
    TileType.WALL:   "█ ",
    TileType.DOOR:   "🚪",
    TileType.KEY:    "🔑",
    TileType.HAZARD: "🔥",
    TileType.GOAL:   "🏆",
}

_DIR_ARROW: Dict[Direction, str] = {
    Direction.NORTH: "▲ ",
    Direction.EAST:  "► ",
    Direction.SOUTH: "▼ ",
    Direction.WEST:  "◄ ",
}

_FOG_CHAR: str = "░ "

# Per-tile Rich styles
_TILE_STYLE: Dict[int, str] = {
    TileType.EMPTY:  "bright_black",
    TileType.WALL:   "white",
    TileType.KEY:    "yellow",
    TileType.DOOR:   "red",
    TileType.HAZARD: "bold red",
    TileType.GOAL:   "bold gold1",
}


class TerminalDashboard:
    """Compact Rich-based terminal renderer.

    Guarantees a strict maximum rendered height of 18 lines.

    Attributes:
        log_buffer: Fixed-length deque of recent timestamped log lines.
    """

    def __init__(self, max_log_lines: int = _THOUGHT_LOG_LINES) -> None:
        self.log_buffer: collections.deque[str] = collections.deque(
            maxlen=max_log_lines,
        )

    # ────────────────────────────────────────────
    #  Public helpers
    # ────────────────────────────────────────────

    def add_log(self, message: str) -> None:
        """Append a timestamped log line to the internal buffer.

        Args:
            message: Free-form log string (timestamp is prepended
                     automatically).
        """
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        idx = len(self.log_buffer) + 1
        self.log_buffer.append(f"[dim]{ts}[/dim] [{idx:02d}] {message}")

    # ────────────────────────────────────────────
    #  Panel builders
    # ────────────────────────────────────────────

    def _render_world_grid(
        self,
        full_grid: np.ndarray,
        explored_grid: np.ndarray,
        player_pos: Tuple[int, int],
        player_dir: Direction,
    ) -> Panel:
        """Render the 10×10 world map with fog-of-war masking.

        Args:
            full_grid:     (10, 10) int32 array of raw TileType values.
            explored_grid: (10, 10) bool array — True where explored.
            player_pos:    (row, col) player position.
            player_dir:    Current facing direction.

        Returns:
            A ``rich.panel.Panel`` containing the styled grid string.
        """
        text = Text()
        rows, cols = full_grid.shape

        for r in range(rows):
            for c in range(cols):
                if (r, c) == player_pos:
                    arrow = _DIR_ARROW.get(player_dir, "? ")
                    text.append(arrow, style="bold cyan")
                elif not explored_grid[r, c]:
                    text.append(_FOG_CHAR, style="dim")
                else:
                    tile_val = int(full_grid[r, c])
                    icon = _TILE_ICON.get(tile_val, "? ")
                    style = _TILE_STYLE.get(tile_val, "white")
                    text.append(icon, style=style)
            # No trailing newline on the last row
            if r < rows - 1:
                text.append("\n")

        return Panel(
            text,
            title="[bold cyan]⚔ WORLD[/bold cyan]",
            border_style="cyan",
            padding=(0, 0),
        )

    def _render_memory_logic_panel(
        self,
        fast_mem_info: Optional[Dict[str, Any]] = None,
        logic_info: Optional[Dict[str, Any]] = None,
    ) -> Panel:
        """Render the Fast Memory & Symbolic Logic telemetry panel.

        Args:
            fast_mem_info: Dict with keys like ``active_goal``,
                ``faiss_match``, ``faiss_distance``, ``loop_detected``.
            logic_info: Dict with keys like ``active_rules``,
                ``safe_to_step``, ``forbidden_actions``.

        Returns:
            A styled ``rich.panel.Panel``.
        """
        mem = fast_mem_info or {}
        logic = logic_info or {}
        t = Text()

        # ── Fast Memory (4 lines) ──
        t.append("━ Memory ━\n", style="bold magenta")

        goal = mem.get("active_goal", "—")
        t.append(" Goal  : ", style="dim")
        t.append(f"{goal}\n", style="bold white")

        faiss_match = mem.get("faiss_match", "—")
        faiss_dist = mem.get("faiss_distance", "—")
        t.append(" FAISS : ", style="dim")
        t.append(f"{faiss_match}", style="bold white")
        if faiss_dist != "—":
            t.append(f" d={faiss_dist:.3f}\n", style="yellow")
        else:
            t.append("\n")

        loop = mem.get("loop_detected", False)
        buf_len = mem.get("buffer_length", 0)
        t.append(" Loop  : ", style="dim")
        if loop:
            t.append("⚠ YES", style="bold red")
        else:
            t.append("No", style="green")
        t.append(f"  Buf:{buf_len}/20\n", style="dim")

        # ── Symbolic Logic (4 lines) ──
        t.append("━ Logic ━\n", style="bold blue")

        safe = logic.get("safe_to_step", "—")
        t.append(" Safe  : ", style="dim")
        if safe is True:
            t.append("True\n", style="bold green")
        elif safe is False:
            t.append("False\n", style="bold red")
        else:
            t.append(f"{safe}\n", style="white")

        rules = logic.get("active_rules", [])
        t.append(" Rules : ", style="dim")
        if rules:
            t.append(", ".join(str(r) for r in rules[:3]) + "\n", style="white")
        else:
            t.append("—\n", style="white")

        forbidden = logic.get("forbidden_actions", [])
        t.append(" Block : ", style="dim")
        if forbidden:
            t.append(", ".join(str(a) for a in forbidden), style="red")
        else:
            t.append("None", style="green")

        return Panel(
            t,
            title="[bold magenta]🧠 MEM+LOGIC[/bold magenta]",
            border_style="magenta",
            padding=(0, 0),
        )

    def _render_thought_stream(self) -> Panel:
        """Render the compact thought-stream log panel.

        Displays at most ``_THOUGHT_LOG_LINES`` (3) entries.

        Returns:
            A ``rich.panel.Panel`` with the most recent log entries.
        """
        text = Text()
        if not self.log_buffer:
            text.append(" (no logs yet)", style="dim italic")
        else:
            entries = list(self.log_buffer)[-_THOUGHT_LOG_LINES:]
            for i, line in enumerate(entries):
                text.append_text(Text.from_markup(line))
                if i < len(entries) - 1:
                    text.append("\n")

        return Panel(
            text,
            title="[bold green]💭 LOG[/bold green]",
            border_style="green",
            padding=(0, 0),
        )

    def _render_status_bar(
        self,
        player_hp: int,
        step_count: int,
        inventory: List[str],
        state: str,
    ) -> Text:
        """Render the single-row horizontal status bar.

        Returns:
            A bare ``rich.text.Text`` line (no borders) to save vertical space.
        """
        # HP colouring
        if player_hp > 60:
            hp_style = "bold green"
        elif player_hp > 30:
            hp_style = "bold yellow"
        else:
            hp_style = "bold red"

        inv_str = ",".join(inventory) if inventory else "∅"

        row = Text()
        row.append(f" ♥{player_hp} ", style=hp_style)
        row.append(f" │ ⏱{step_count} ", style="bold white")
        row.append(f" │ 🎒[{inv_str}] ", style="bold yellow")
        row.append(f" │ ⚙ {state}", style="bold cyan")

        return row

    # ────────────────────────────────────────────
    #  Layout assembler
    # ────────────────────────────────────────────

    def generate_layout(
        self,
        obs_dict: Dict[str, Any],
        fast_mem_info: Optional[Dict[str, Any]] = None,
        logic_info: Optional[Dict[str, Any]] = None,
        step_count: int = 0,
        engine_state: str = "EXPLORING",
    ) -> Layout:
        """Assemble the strict 18-line Rich Layout.

        Args:
            obs_dict:     Observation dict from ``CustomRPGEnv`` containing
                          ``fov``, ``full_grid``, and ``player_state``.
            fast_mem_info: Optional telemetry dict from Fast Memory.
            logic_info:   Optional telemetry dict from Symbolic Engine.
            step_count:   Current step counter.
            engine_state: High-level engine state label.

        Returns:
            ``rich.layout.Layout`` explicitly constrained to 18 lines.
        """
        ps = obs_dict.get("player_state", {})
        player_pos: Tuple[int, int] = ps.get("position", (0, 0))
        player_dir: Direction = ps.get("direction", Direction.NORTH)
        player_hp: int = ps.get("health", 0)
        inventory: List[str] = ps.get("inventory", [])

        # Extract grids — rebuild explored mask from full_grid fog values
        full_grid: np.ndarray = obs_dict.get(
            "full_grid", np.zeros((10, 10), dtype=np.int32),
        )
        explored_grid: np.ndarray = full_grid >= 0  # FOG == -1

        # ── Build panels ──
        world_panel = self._render_world_grid(
            full_grid, explored_grid, player_pos, player_dir,
        )
        mem_logic_panel = self._render_memory_logic_panel(
            fast_mem_info, logic_info,
        )
        thought_panel = self._render_thought_stream()
        status_line = self._render_status_bar(
            player_hp, step_count, inventory, engine_state,
        )

        # ── Layout assembly (fixed sizes) ──
        #
        #  ┌─── top (12 lines) ─────────────────┐
        #  │  left (world)  │  right (mem)       │
        #  │  size=12       │  size=12           │
        #  ├─── thought (5 lines) ──────────────┤
        #  │  thought stream                     │
        #  ├─── status (1 line) ────────────────┤
        #  │  status bar (Text)                  │
        #  └─────────────────────────────────────┘

        layout = Layout(name="root", size=_MAX_HEIGHT)
        layout.split_column(
            Layout(name="top", size=_TOP_HEIGHT),
            Layout(name="thought", size=_THOUGHT_HEIGHT),
            Layout(name="status", size=_STATUS_HEIGHT),
        )
        layout["top"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        layout["left"].update(world_panel)
        layout["right"].update(mem_logic_panel)
        layout["thought"].update(thought_panel)
        layout["status"].update(status_line)

        return layout
