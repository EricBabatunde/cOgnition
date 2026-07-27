#!/usr/bin/env python3
"""
main.py — Dual-Mode Cognitive Execution Loop
=================================================
Provides both keyboard-driven manual teleop and fully autonomous
goal-directed execution modes for the Custom RPG Environment
with real-time Rich terminal dashboard rendering.

Manual Controls:
    W / ↑       Move Forward
    A / ←       Turn Left
    D / →       Turn Right
    E           Pick Up Item
    Space       Toggle Interact (Door / Lever)
    Q           Quit

Autonomous Mode:
    --auto      Run hands-off with Executive Goal Synthesis

Target: Python 3.10  |  Platform: Linux (termios)
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rich.console import Console
from rich.live import Live

from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.executive_admin import ExecutiveGoalEngine
from cognitive.fast_memory import FastPlasticityMemory
from cognitive.symbolic_engine import SymbolicLogicEngine
from environment import Action, CustomRPGEnv, Direction
from environment.entities import FOG, TileType
from ui.terminal_dashboard import TerminalDashboard
from ui.web_inspector import WebMindInspector

# Direction enum → direction vector mapping
_DIR_TO_VEC: Dict[Direction, Tuple[int, int]] = {
    Direction.NORTH: (-1, 0),
    Direction.EAST:  (0, 1),
    Direction.SOUTH: (1, 0),
    Direction.WEST:  (0, -1),
}

# Action name string → Action enum mapping
_ACTION_NAME_MAP: Dict[str, Action] = {a.name: a for a in Action}


# ──────────────────────────────────────────────
# Terminal Mode Management
# ──────────────────────────────────────────────

def set_raw_mode(fd: int) -> list:
    """Set the terminal to non-canonical (cbreak) mode with no echo.

    Returns:
        The original termios settings to be restored later.
    """
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    # Explicitly turn off ECHO if setcbreak didn't
    new_settings = termios.tcgetattr(fd)
    new_settings[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

    return old_settings


# ──────────────────────────────────────────────
# Non-blocking Input Poller
# ──────────────────────────────────────────────

def get_action_key(fd: int, timeout: float = 0.02) -> Optional[Union[Action, str]]:
    """Poll stdin for a valid action key or arrow sequence.

    Reads up to 32 bytes to handle escape sequences. Discards mouse/scroll
    events and unrecognized ANSI codes.

    Returns:
        Action enum, 'QUIT' string, or None if no valid input.
    """
    if not select.select([fd], [], [], timeout)[0]:
        return None

    try:
        data = os.read(fd, 32)
    except BlockingIOError:
        return None

    if not data:
        return None

    # Filter out mouse and trackpad scroll sequences
    if (data.startswith(b'\x1b[M') or
        data.startswith(b'\x1b[<') or
        data.startswith(b'\x1b[35') or
        data.startswith(b'\x1b[36')):
        while select.select([fd], [], [], 0.0)[0]:
            os.read(fd, 1024)
        return None

    try:
        s = data.decode('utf-8', errors='ignore')
    except Exception:
        return None

    if not s:
        return None

    # Exact matches for arrow sequences
    if s == '\x1b[A':
        return Action.MOVE_FORWARD
    if s == '\x1b[D':
        return Action.TURN_LEFT
    if s == '\x1b[C':
        return Action.TURN_RIGHT

    ch = s[0]

    if ch in ('w', 'W'):
        return Action.MOVE_FORWARD
    if ch in ('a', 'A'):
        return Action.TURN_LEFT
    if ch in ('d', 'D'):
        return Action.TURN_RIGHT
    if ch in ('e', 'E'):
        return Action.PICK_UP
    if ch == ' ':
        return Action.TOGGLE_INTERACT
    if ch in ('q', 'Q', '\x03'):
        return 'QUIT'

    return None


# ──────────────────────────────────────────────
# Direction label helper
# ──────────────────────────────────────────────

_DIR_SYMBOL: dict[Direction, str] = {
    Direction.NORTH: "▲ N",
    Direction.EAST:  "► E",
    Direction.SOUTH: "▼ S",
    Direction.WEST:  "◄ W",
}

# ──────────────────────────────────────────────
# TileType → entity type string mapping
# ──────────────────────────────────────────────

_TILE_ENTITY_TYPE: Dict[int, str] = {
    TileType.KEY:    "KEY",
    TileType.DOOR:   "DOOR",
    TileType.HAZARD: "HAZARD",
    TileType.GOAL:   "GOAL",
}

_TILE_ENTITY_PREFIX: Dict[int, str] = {
    TileType.KEY:    "Key",
    TileType.DOOR:   "Door",
    TileType.HAZARD: "Hazard",
    TileType.GOAL:   "Goal",
}


# ──────────────────────────────────────────────
# Target Tile & Grounding Helper
# ──────────────────────────────────────────────

def get_forward_tile_context(env: CustomRPGEnv, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract grounding facts for the tile immediately in front of the agent."""
    ps = obs["player_state"]
    px, py = ps["position"]
    direction = Direction(ps["direction"])

    dx, dy = 0, 0
    if direction == Direction.NORTH:
        dx, dy = -1, 0
    elif direction == Direction.SOUTH:
        dx, dy = 1, 0
    elif direction == Direction.EAST:
        dx, dy = 0, 1
    elif direction == Direction.WEST:
        dx, dy = 0, -1

    fx, fy = px + dx, py + dy

    grid = env.unwrapped._grid
    if 0 <= fx < grid.shape[0] and 0 <= fy < grid.shape[1]:
        tile_val = int(grid[fx, fy])
    else:
        tile_val = int(TileType.WALL)

    try:
        tile_type = TileType(tile_val)
    except ValueError:
        tile_type = TileType.EMPTY

    door_is_locked = False
    if tile_type == TileType.DOOR:
        entity = env.unwrapped._entities.get((fx, fy))
        if entity:
            door_is_locked = entity.is_locked

    has_key = any("key" in item.lower() for item in ps.get("inventory", []))

    return {
        "is_wall": tile_type == TileType.WALL,
        "is_hazard": tile_type == TileType.HAZARD,
        "is_door": tile_type == TileType.DOOR,
        "is_locked": tile_type == TileType.DOOR and door_is_locked,
        "has_key": has_key,
        "target_tile": tile_type.name,
    }


# ──────────────────────────────────────────────
# Metacognitive Reflection & Consolidation
# ──────────────────────────────────────────────

def trigger_sleep_consolidation(
    fast_memory: FastPlasticityMemory, 
    matrix: CoreKnowledgeMatrix, 
    dashboard: TerminalDashboard,
    novelty: float
) -> None:
    """Offline reflection cycle triggered by high prediction error."""
    recent_experiences = fast_memory.get_recent_experiences(5)
    
    for exp in recent_experiences:
        ps = exp.state_dict
        px, py = ps.get("position", (0, 0))
        # Ensure corresponding topological nodes exist in CoreKnowledgeMatrix
        matrix.add_spatial_node(px, py, "EMPTY", explored=True)
        
    dashboard.add_log(f"🧠 [REFLECTION CYCLE] High Prediction Error (ΔE={novelty:.2f}) -> Consolidated {len(recent_experiences)} RAM nodes into Core Graph")


# ──────────────────────────────────────────────
# Perception-to-Graph Helper
# ──────────────────────────────────────────────

def update_knowledge_from_obs(
    matrix: CoreKnowledgeMatrix,
    obs: Dict[str, Any],
    prev_pos: Optional[Tuple[int, int]] = None,
) -> None:
    """Synchronise the knowledge graph with the latest observation.

    Grounds the agent's current position, explored FOV tiles,
    visible entities, and inventory into the ``CoreKnowledgeMatrix``.

    Args:
        matrix:   The knowledge graph to update.
        obs:      Observation dict from ``CustomRPGEnv``.
        prev_pos: Player position before the latest step (for
                  topological edge creation). ``None`` on reset.
    """
    ps = obs["player_state"]
    px, py = ps["position"]
    full_grid: np.ndarray = obs["full_grid"]

    # ── 1. Current position spatial node ──
    matrix.add_spatial_node(px, py, "EMPTY", explored=True)

    # ── 2. Topological edge from previous position ──
    if prev_pos is not None and prev_pos != (px, py):
        prev_r, prev_c = prev_pos
        matrix.add_spatial_node(prev_r, prev_c, "EMPTY", explored=True)
        src = f"Tile_{prev_r}_{prev_c}"
        dst = f"Tile_{px}_{py}"
        # Bidirectional connectivity
        matrix.add_typed_edge(src, dst, "CONNECTS_TO")
        matrix.add_typed_edge(dst, src, "CONNECTS_TO")

    # ── 3. FOV grounding from full_grid (explored tiles) ──
    rows, cols = full_grid.shape
    for r in range(rows):
        for c in range(cols):
            tile_val = int(full_grid[r, c])
            if tile_val == FOG:
                continue  # unexplored

            # Determine tile type string for spatial node
            try:
                tile_type = TileType(tile_val)
            except ValueError:
                continue

            tile_name = tile_type.name  # e.g. "EMPTY", "WALL", etc.
            matrix.add_spatial_node(r, c, tile_name, explored=True)

            # Link entity nodes for notable tile types
            if tile_val in _TILE_ENTITY_TYPE:
                prefix = _TILE_ENTITY_PREFIX[tile_val]
                entity_id = f"{prefix}_{r}_{c}"
                entity_type = _TILE_ENTITY_TYPE[tile_val]
                matrix.add_entity_node(entity_id, entity_type, {
                    "location": f"({r},{c})",
                })
                host_tile = f"Tile_{r}_{c}"
                matrix.add_typed_edge(host_tile, entity_id, "CONTAINS")

    # ── 4. Spatial connectivity for explored adjacent tiles ──
    for r in range(rows):
        for c in range(cols):
            if int(full_grid[r, c]) == FOG:
                continue
            tile_val = int(full_grid[r, c])
            # Walls don't connect
            if tile_val == TileType.WALL:
                continue
            src_id = f"Tile_{r}_{c}"
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbour_val = int(full_grid[nr, nc])
                    if neighbour_val == FOG or neighbour_val == TileType.WALL:
                        continue
                    dst_id = f"Tile_{nr}_{nc}"
                    matrix.add_typed_edge(src_id, dst_id, "CONNECTS_TO")

    # ── 5. Inventory grounding ──
    # Ensure Agent node exists
    matrix.add_entity_node("Agent", "PLAYER", {
        "position": f"({px},{py})",
        "direction": Direction(ps["direction"]).name,
        "health": ps["health"],
    })
    for item_id in ps.get("inventory", []):
        matrix.add_entity_node(item_id, "ITEM", {"held": True})
        matrix.add_typed_edge("Agent", item_id, "IN_INVENTORY")


# ──────────────────────────────────────────────
# Autonomous Goal-Directed Execution Loop
# ──────────────────────────────────────────────

def run_autonomous_loop(
    env: CustomRPGEnv,
    matrix: CoreKnowledgeMatrix,
    symbolic_engine: SymbolicLogicEngine,
    fast_memory: FastPlasticityMemory,
    goal_engine: ExecutiveGoalEngine,
    dash: TerminalDashboard,
    inspector: WebMindInspector,
    console: Console,
    max_steps: int = 200,
    render_dashboard: bool = True,
) -> Dict[str, Any]:
    """Fully autonomous cognitive execution loop.

    Integrates Executive Goal Synthesis, A* Pathfinding,
    Z3 Safety Verification, and Fast RAM Reflection into
    a unified hands-off event loop.

    Returns:
        Telemetry dict with performance metrics.
    """
    obs, info = env.reset(seed=42)
    ps = obs["player_state"]
    update_knowledge_from_obs(matrix, obs, prev_pos=None)

    # ── Telemetry counters ──
    step_count = 0
    safety_blocks = 0
    reflection_cycles = 0
    current_novelty = 0.0
    current_weight = 1.0
    engine_state = "AUTONOMOUS"
    action_queue: List[str] = []

    t_start = time.perf_counter()

    dash.add_log("[bold cyan]⚡ Autonomous mode engaged[/bold cyan]")

    terminated = False
    truncated = False

    def _refresh_dashboard() -> None:
        if not render_dashboard:
            return
        return dash.generate_layout(
            obs_dict=obs,
            step_count=step_count,
            engine_state=engine_state,
            fast_mem_info={
                "faiss_count": fast_memory.faiss_index.ntotal,
                "novelty_score": current_novelty,
                "active_weight": current_weight,
            },
        )

    with Live(
        _refresh_dashboard(),
        console=console,
        refresh_per_second=15,
        auto_refresh=True,
        screen=render_dashboard,
    ) as live:

        for tick in range(max_steps):
            if terminated or truncated or ps["health"] <= 0:
                break

            prev_pos = tuple(ps["position"])

            # ── Step 1: Ground state context & novelty ──
            state_context = get_forward_tile_context(env, obs)
            fast_memory.step_clock()
            vec = fast_memory.vectorizer.vectorize(obs, state_context)
            current_novelty = fast_memory.calculate_novelty(vec)

            # ── Step 2: Reflection trigger on high ΔE ──
            if current_novelty >= 0.50:
                trigger_sleep_consolidation(
                    fast_memory, matrix, dash, current_novelty
                )
                reflection_cycles += 1
                action_queue.clear()  # Invalidate stale plan

            # ── Step 3: Plan maintenance ──
            if not action_queue:
                direction = Direction(ps["direction"])
                current_dir = _DIR_TO_VEC[direction]
                inventory = ps.get("inventory", [])
                pos = tuple(ps["position"])

                goal_stack = goal_engine.synthesize_goal_stack(pos, inventory)
                if goal_stack:
                    action_queue = goal_engine.compile_execution_plan(
                        goal_stack, pos, current_dir, inventory
                    )
                    goals_str = ", ".join(g.goal_type.name for g in goal_stack)
                    dash.add_log(
                        f"[bold magenta]📋 Plan:[/bold magenta] {goals_str} "
                        f"({len(action_queue)} actions)"
                    )
                else:
                    # Fallback: random exploratory action
                    action_queue = [Action(env.action_space.sample()).name]

            # ── Step 4: Dispatch next action with safety gate ──
            act_name = action_queue.pop(0)
            action = _ACTION_NAME_MAP.get(act_name)
            if action is None:
                continue

            # Z3 safety interception for forward movement
            if action == Action.MOVE_FORWARD:
                state_context = get_forward_tile_context(env, obs)
                is_safe, explanation, status, active_rules = (
                    symbolic_engine.verify_action_dynamic(
                        "MOVE_FORWARD", state_context
                    )
                )
                if not is_safe:
                    safety_blocks += 1
                    action_queue.clear()  # Purge invalid plan
                    dash.add_log(
                        f"[bold red]⛔ Z3 BLOCK:[/bold red] {explanation}"
                    )
                    if render_dashboard:
                        live.update(_refresh_dashboard())
                    continue

            # ── Step 5: Execute valid action ──
            obs, reward, terminated, truncated, info = env.step(action)
            ps = obs["player_state"]
            step_count = info["step_count"]

            # Cache experience & reinforce
            exp = fast_memory.store_experience(
                obs, state_context, action.name, reward
            )
            current_weight = exp.weight
            if reward > 0:
                tile = state_context.get("target_tile", "EMPTY")
                fast_memory.reinforce(tile, action.name, reward)

            update_knowledge_from_obs(matrix, obs, prev_pos=prev_pos)

            # ── Step 6: Terminal detection ──
            hp = ps["health"]
            if terminated and hp > 0:
                engine_state = "GOAL_REACHED"
                dash.add_log(
                    f"[bold green]🏆 GOAL![/bold green] "
                    f"Steps:{step_count} HP:{hp}"
                )
            elif terminated or hp <= 0:
                engine_state = "GAME_OVER"
                dash.add_log(
                    f"[bold red]💀 GAME OVER[/bold red] step {step_count}"
                )
            elif truncated:
                engine_state = "TRUNCATED"

            if render_dashboard:
                live.update(_refresh_dashboard())
            time.sleep(0.02)

    elapsed = time.perf_counter() - t_start

    telemetry = {
        "total_steps": step_count,
        "safety_blocks": safety_blocks,
        "reflection_cycles": reflection_cycles,
        "elapsed_seconds": elapsed,
        "engine_state": engine_state,
        "final_hp": ps["health"],
        "faiss_count": fast_memory.faiss_index.ntotal,
    }
    return telemetry


# ──────────────────────────────────────────────
# Manual interactive loop
# ──────────────────────────────────────────────

def run_manual_loop() -> int:
    """Run the interactive teleop loop."""
    console = Console()
    env = CustomRPGEnv()
    dash = TerminalDashboard()
    matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
    inspector = WebMindInspector(matrix)

    symbolic_engine = SymbolicLogicEngine()
    fast_memory = FastPlasticityMemory(dimension=64, capacity=1000)
    symbolic_engine.load_rules_from_config("config/innate_instincts.json")

    obs, info = env.reset(seed=42)
    ps = obs["player_state"]

    # Initialize UI state tracking variables for fast memory
    current_novelty = 0.0
    current_weight = 1.0

    # Initial knowledge grounding
    update_knowledge_from_obs(matrix, obs, prev_pos=None)

    summary = matrix.get_graph_summary()
    dash.add_log("[bold cyan]Engine init[/bold cyan] — spawn (1,1) EAST")
    dash.add_log(
        f"[dim]Graph: {summary['total_nodes']}N "
        f"{summary['total_edges']}E[/dim]"
    )

    terminated = False
    truncated = False
    step_count = 0
    engine_state = "EXPLORING"

    fd = sys.stdin.fileno()
    original_term = set_raw_mode(fd)

    try:
        with Live(
            dash.generate_layout(
                obs_dict=obs,
                step_count=step_count,
                engine_state=engine_state,
                fast_mem_info={
                    "faiss_count": fast_memory.faiss_index.ntotal,
                    "novelty_score": current_novelty,
                    "active_weight": current_weight
                }
            ),
            console=console,
            refresh_per_second=15,
            auto_refresh=True,
            screen=True,
        ) as live:

            while not terminated and not truncated and ps["health"] > 0:
                action_or_quit = get_action_key(fd, timeout=0.02)

                if action_or_quit == 'QUIT':
                    dash.add_log("[bold yellow]⏹ Quit[/bold yellow]")
                    live.update(dash.generate_layout(
                        obs_dict=obs,
                        step_count=step_count,
                        engine_state="QUIT",
                    ))
                    break

                elif isinstance(action_or_quit, Action):
                    action = action_or_quit
                    prev_pos = tuple(ps["position"])

                    # ── 1. Fast Memory Step Clock & Novelty Grounding ──
                    state_context = get_forward_tile_context(env, obs)
                    fast_memory.step_clock()
                    vec = fast_memory.vectorizer.vectorize(obs, state_context)
                    current_novelty = fast_memory.calculate_novelty(vec)

                    # ── 2. Executive Safety Gate Interception ──
                    if action == Action.MOVE_FORWARD:
                        is_safe, explanation, status, active_rules = symbolic_engine.verify_action_dynamic(
                            "MOVE_FORWARD", state_context
                        )
                        if not is_safe:
                            dash.add_log(
                                f"[bold red]BLOCKED [Z3 UNSAT]:[/bold red] "
                                f"{explanation} | Rules: {active_rules}"
                            )
                            live.update(dash.generate_layout(
                                obs_dict=obs,
                                step_count=step_count,
                                engine_state=engine_state,
                                fast_mem_info={
                                    "faiss_count": fast_memory.faiss_index.ntotal,
                                    "novelty_score": current_novelty,
                                    "active_weight": current_weight
                                }
                            ))
                            time.sleep(0.01)
                            continue

                    # ── 3. Execute Valid Action ──
                    obs, reward, terminated, truncated, info = env.step(action)
                    ps = obs["player_state"]
                    step_count = info["step_count"]

                    # ── 4. Cache Episodic Experience ──
                    exp = fast_memory.store_experience(obs, state_context, action.name, reward)
                    current_weight = exp.weight

                    # ── 5. Metacognitive Reflection Trigger ──
                    if current_novelty >= 0.50:
                        trigger_sleep_consolidation(fast_memory, matrix, dash, current_novelty)

                    # ── Update knowledge graph ──
                    update_knowledge_from_obs(matrix, obs, prev_pos=prev_pos)

                    # ── Build log entry ──
                    pos = ps["position"]
                    direction = ps["direction"]
                    dir_label = _DIR_SYMBOL.get(Direction(direction), "?")
                    hp = ps["health"]

                    wall_blocked = (
                        action == Action.MOVE_FORWARD
                        and tuple(pos) == prev_pos
                    )

                    if wall_blocked:
                        dash.add_log(
                            f"Action: [bold]{action.name}[/bold]"
                            f" | Reward: {reward:.1f}"
                            f" | [bold red]Wall Blocked[/bold red]"
                        )
                    else:
                        log_parts = [
                            f"[bold]{action.name}[/bold]",
                            f"→ ({pos[0]},{pos[1]}) {dir_label}",
                            f"HP:{hp}",
                        ]

                        if reward != -0.1:
                            log_parts.append(
                                f"[bold yellow]R={reward:+.1f}[/bold yellow]"
                            )

                        if ps["inventory"]:
                            log_parts.append(f"🎒 {ps['inventory']}")

                        dash.add_log("  ".join(log_parts))

                    # ── Detect special events ──
                    if terminated and hp > 0:
                        engine_state = "GOAL_REACHED"
                        dash.add_log(
                            "[bold green]🏆 GOAL![/bold green]"
                            f" Steps:{step_count} HP:{hp}"
                        )
                    elif terminated or hp <= 0:
                        engine_state = "GAME_OVER"
                        dash.add_log(
                            "[bold red]💀 GAME OVER[/bold red]"
                            f" step {step_count}"
                        )
                    elif truncated:
                        engine_state = "TRUNCATED"
                        dash.add_log(
                            "[bold yellow]⏱ Truncated[/bold yellow]"
                            f" step {step_count}"
                        )

                    # ── Refresh display ──
                    live.update(dash.generate_layout(
                        obs_dict=obs,
                        step_count=step_count,
                        engine_state=engine_state,
                        fast_mem_info={
                            "faiss_count": fast_memory.faiss_index.ntotal,
                            "novelty_score": current_novelty,
                            "active_weight": current_weight
                        }
                    ))

                # Prevent 100% CPU utilization
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_term)

    # ── Export knowledge graph ──
    output_path = inspector.render_html("graph_mind.html")
    summary = matrix.get_graph_summary()

    console.print()
    console.rule("[bold cyan]Session Summary[/bold cyan]")
    console.print(f"  [bold]Engine State:[/bold]  {engine_state}")
    console.print(f"  [bold]Total Steps:[/bold]   {step_count}")
    console.print(f"  [bold]Final HP:[/bold]      {ps['health']}")
    console.print(f"  [bold]Inventory:[/bold]     {ps['inventory']}")
    console.print(f"  [bold]Explored:[/bold]      {info['explored_pct'] * 100:.1f}%")
    console.print(
        f"  [bold]Graph:[/bold]        "
        f"{summary['total_nodes']} nodes, "
        f"{summary['total_edges']} edges  "
        f"{summary['node_type_counts']}"
    )
    console.print(f"  [bold green][INFO][/bold green] Knowledge Matrix exported → {output_path}")
    console.rule()
    console.print()

    return 0


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> int:
    """Dual-mode entry point: --auto for autonomous, default for manual."""
    if "--auto" in sys.argv:
        console = Console()
        env = CustomRPGEnv()
        dash = TerminalDashboard()
        matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
        inspector = WebMindInspector(matrix)
        symbolic_engine = SymbolicLogicEngine()
        symbolic_engine.load_rules_from_config("config/innate_instincts.json")
        fast_memory = FastPlasticityMemory(dimension=64, capacity=1000)
        goal_engine = ExecutiveGoalEngine(matrix, symbolic_engine)

        telemetry = run_autonomous_loop(
            env=env,
            matrix=matrix,
            symbolic_engine=symbolic_engine,
            fast_memory=fast_memory,
            goal_engine=goal_engine,
            dash=dash,
            inspector=inspector,
            console=console,
            max_steps=200,
            render_dashboard=True,
        )

        output_path = inspector.render_html("graph_mind.html")
        summary = matrix.get_graph_summary()

        console.print()
        console.rule("[bold cyan]Autonomous Session Summary[/bold cyan]")
        console.print(f"  [bold]Engine State:[/bold]    {telemetry['engine_state']}")
        console.print(f"  [bold]Total Steps:[/bold]     {telemetry['total_steps']}")
        console.print(f"  [bold]Final HP:[/bold]        {telemetry['final_hp']}")
        console.print(f"  [bold]Safety Blocks:[/bold]   {telemetry['safety_blocks']}")
        console.print(f"  [bold]Reflections:[/bold]     {telemetry['reflection_cycles']}")
        console.print(f"  [bold]FAISS Vectors:[/bold]   {telemetry['faiss_count']}")
        console.print(f"  [bold]Elapsed:[/bold]         {telemetry['elapsed_seconds']:.2f}s")
        console.print(
            f"  [bold]Graph:[/bold]          "
            f"{summary['total_nodes']} nodes, "
            f"{summary['total_edges']} edges  "
            f"{summary['node_type_counts']}"
        )
        console.print(f"  [bold green][INFO][/bold green] Knowledge Matrix exported → {output_path}")
        console.rule()
        console.print()
        return 0

    return run_manual_loop()


if __name__ == "__main__":
    sys.exit(main())
