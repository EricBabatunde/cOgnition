"""
terminal_dashboard.py — Rich Split-Panel Terminal Dashboard
============================================================
Renders a 3-panel + status-bar terminal UI for the Neuro-Symbolic
RPG Cognitive Engine using the ``rich`` library.

Layout
------
::

    ┌─────────────────────┬───────────────────────┐
    │                     │  FAST MEMORY & LOGIC   │
    │    WORLD VIEW       │  - Active Goal         │
    │   (Fog of War)      │  - FAISS Match Dist    │
    │                     │  - Active Rules        │
    │                     ├───────────────────────┤
    │                     │  THOUGHT STREAM LOG    │
    │                     │  [01] …                │
    │                     │  [02] …                │
    ├─────────────────────┴───────────────────────┤
    │  HP: 100 │ Step: 14 │ Inv: [Key_Red] │ …    │
    └─────────────────────────────────────────────┘

Target: Python 3.10
"""

from __future__ import annotations

import collections
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from environment.entities import Direction, TileType

# ──────────────────────────────────────────────
# Tile rendering constants
# ──────────────────────────────────────────────
_TILE_ICON: Dict[int, str] = {
    TileType.EMPTY:  ". ",
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

_FOG_CHAR: str = "░░"


class TerminalDashboard:
    """Rich-based split-panel terminal renderer.

    Assembles a ``rich.layout.Layout`` containing:

    * **World View** — 10×10 fog-of-war grid with entity icons.
    * **Fast Memory & Logic** — subsystem telemetry panel.
    * **Thought Stream** — rolling log of engine decisions.
    * **Status Bar** — HP / Step / Inventory / Engine State.

    Attributes:
        log_buffer: Fixed-length deque of recent timestamped log lines.
    """

    def __init__(self, max_log_lines: int = 10) -> None:
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
        self.log_buffer.append(f"[dim]{ts}[/dim] [bold][{idx:02d}][/bold] {message}")

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
                    # Per-type styling
                    if tile_val == TileType.WALL:
                        text.append(icon, style="white")
                    elif tile_val == TileType.KEY:
                        text.append(icon, style="yellow")
                    elif tile_val == TileType.DOOR:
                        text.append(icon, style="red")
                    elif tile_val == TileType.HAZARD:
                        text.append(icon, style="bold red")
                    elif tile_val == TileType.GOAL:
                        text.append(icon, style="bold gold1")
                    else:
                        text.append(icon, style="bright_black")
            text.append("\n")

        return Panel(
            text,
            title="[bold cyan]⚔  WORLD VIEW[/bold cyan]",
            subtitle="[dim]Fog of War 10×10[/dim]",
            border_style="cyan",
            padding=(0, 1),
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

        lines = Text()

        # Fast Memory section
        lines.append("━ Fast Memory ━\n", style="bold magenta")
        goal = mem.get("active_goal", "—")
        lines.append(f"  Active Goal   : ", style="dim")
        lines.append(f"{goal}\n", style="bold white")

        faiss_match = mem.get("faiss_match", "—")
        faiss_dist = mem.get("faiss_distance", "—")
        lines.append(f"  FAISS Match   : ", style="dim")
        lines.append(f"{faiss_match}", style="bold white")
        if faiss_dist != "—":
            lines.append(f"  (d={faiss_dist:.4f})\n", style="yellow")
        else:
            lines.append("\n")

        loop = mem.get("loop_detected", False)
        lines.append(f"  Loop Detected : ", style="dim")
        if loop:
            lines.append("⚠ YES\n", style="bold red")
        else:
            lines.append("No\n", style="green")

        buffer_len = mem.get("buffer_length", 0)
        lines.append(f"  Buffer Depth  : ", style="dim")
        lines.append(f"{buffer_len}/20\n", style="white")

        # Symbolic Logic section
        lines.append("\n")
        lines.append("━ Symbolic Logic ━\n", style="bold blue")

        safe = logic.get("safe_to_step", "—")
        lines.append(f"  SafeToStep    : ", style="dim")
        if safe is True:
            lines.append("True\n", style="bold green")
        elif safe is False:
            lines.append("False\n", style="bold red")
        else:
            lines.append(f"{safe}\n", style="white")

        rules = logic.get("active_rules", [])
        lines.append(f"  Active Rules  : ", style="dim")
        if rules:
            lines.append(f"{len(rules)}\n", style="white")
            for rule in rules[:4]:
                lines.append(f"    • {rule}\n", style="dim")
        else:
            lines.append("—\n", style="white")

        forbidden = logic.get("forbidden_actions", [])
        lines.append(f"  Forbidden     : ", style="dim")
        if forbidden:
            lines.append(", ".join(str(a) for a in forbidden) + "\n", style="red")
        else:
            lines.append("None\n", style="green")

        return Panel(
            lines,
            title="[bold magenta]🧠 MEMORY & LOGIC[/bold magenta]",
            border_style="magenta",
            padding=(0, 1),
        )

    def _render_thought_stream(self) -> Panel:
        """Render the scrolling thought-stream log panel.

        Returns:
            A ``rich.panel.Panel`` with the most recent log entries.
        """
        text = Text()
        if not self.log_buffer:
            text.append("  (no log entries yet)\n", style="dim italic")
        else:
            for line in self.log_buffer:
                text.append_text(Text.from_markup(line + "\n"))

        return Panel(
            text,
            title="[bold green]💭 THOUGHT STREAM[/bold green]",
            border_style="green",
            padding=(0, 1),
        )

    def _render_status_bar(
        self,
        player_hp: int,
        step_count: int,
        inventory: List[str],
        state: str,
    ) -> Panel:
        """Render the horizontal status bar.

        Args:
            player_hp:  Current hit-points.
            step_count: Total environment steps elapsed.
            inventory:  List of item_id strings in the player's bag.
            state:      Engine state label (e.g. EXPLORING, REFLECTING).

        Returns:
            A single-line ``rich.panel.Panel``.
        """
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="left", ratio=2)
        table.add_column(justify="right", ratio=1)

        # HP colouring
        if player_hp > 60:
            hp_style = "bold green"
        elif player_hp > 30:
            hp_style = "bold yellow"
        else:
            hp_style = "bold red"

        hp_text = Text(f"♥ HP: {player_hp}", style=hp_style)
        step_text = Text(f"⏱ Step: {step_count}", style="bold white")

        inv_str = ", ".join(inventory) if inventory else "∅"
        inv_text = Text(f"🎒 Inv: [{inv_str}]", style="bold yellow")

        state_text = Text(f"⚙ {state}", style="bold cyan")

        table.add_row(hp_text, step_text, inv_text, state_text)

        return Panel(
            table,
            border_style="bright_black",
            padding=(0, 1),
        )

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
        """Assemble the full 3-panel + status-bar Rich Layout.

        Args:
            obs_dict:     Observation dict from ``CustomRPGEnv`` containing
                          ``fov``, ``full_grid``, and ``player_state``.
            fast_mem_info: Optional telemetry dict from Fast Memory.
            logic_info:   Optional telemetry dict from Symbolic Engine.
            step_count:   Current step counter.
            engine_state: High-level engine state label.

        Returns:
            ``rich.layout.Layout`` ready for ``Console.print()`` or
            ``Live`` rendering.
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

        # ── Panels ──
        world_panel = self._render_world_grid(
            full_grid, explored_grid, player_pos, player_dir,
        )
        mem_logic_panel = self._render_memory_logic_panel(
            fast_mem_info, logic_info,
        )
        thought_panel = self._render_thought_stream()
        status_panel = self._render_status_bar(
            player_hp, step_count, inventory, engine_state,
        )

        # ── Layout assembly ──
        #
        #  ┌──── body ──────────────────────────────┐
        #  │  left (world)  │  right                 │
        #  │                │  ┌─ right_top ───────┐ │
        #  │                │  │ mem & logic        │ │
        #  │                │  ├─ right_bottom ────┤ │
        #  │                │  │ thought stream    │ │
        #  │                │  └───────────────────┘ │
        #  ├──── footer ────────────────────────────┤
        #  │  status bar                             │
        #  └─────────────────────────────────────────┘

        layout = Layout(name="root")
        layout.split_column(
            Layout(name="body", ratio=5),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        layout["right"].split_column(
            Layout(name="right_top", ratio=3),
            Layout(name="right_bottom", ratio=2),
        )

        layout["left"].update(world_panel)
        layout["right_top"].update(mem_logic_panel)
        layout["right_bottom"].update(thought_panel)
        layout["footer"].update(status_panel)

        return layout
