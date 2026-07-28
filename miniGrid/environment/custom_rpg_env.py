"""
custom_rpg_env.py — Gymnasium-Compatible RPG Grid Environment
=============================================================
Implements a tile-based environment with fog-of-war, item mechanics,
hazard damage, dynamic levers/pressure plates, unmapped traps, and
goal-seeking dynamics for the Neuro-Symbolic Cognitive Engine.

Supports three difficulty tiers:
    Tier 1 — 10×10 single room, 1 key, 1 door, goal chest.
    Tier 2 — 15×15 multi-room, 2 keys, 2 doors, static lava.
    Tier 3 — 20×20 dungeon, hidden traps, levers, decoy chests.

Target: Python 3.10
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from .entities import (
    Action,
    Direction,
    Entity,
    FOG,
    PlayerState,
    TileType,
)

# Direction deltas: (delta_row, delta_col)
_DIR_DELTA: Dict[Direction, Tuple[int, int]] = {
    Direction.NORTH: (-1, 0),
    Direction.EAST:  (0, 1),
    Direction.SOUTH: (1, 0),
    Direction.WEST:  (0, -1),
}

FOV_SIZE: int = 5
INITIAL_HP: int = 100
STEP_PENALTY: float = -0.1
GOAL_REWARD: float = 100.0

# Default step limit — can be overridden at module level by tests
MAX_STEPS: int = 200


class CustomRPGEnv(gymnasium.Env):
    """A tile-based RPG environment with fog-of-war and multi-tier maps.

    Observation dict keys:
        fov          — (5, 5) int32 egocentric vision slice.
        full_grid    — (H, W) int32 world map (FOG = -1 for unexplored).
        player_state — dict with position, direction, health, inventory.

    Reward mechanics:
        Every step       →  -0.1  (step penalty)
        Hazard tile      →  HP reduced by entity damage
        Unmapped trap    →  HP reduced by 20 (surprise damage)
        Goal reached     →  +100, episode terminates
        Decoy chest      →  No reward, no termination
        HP ≤ 0           →  episode terminates (reward = 0)

    Args:
        tier:        Difficulty tier (1, 2, or 3). Default is 1.
        render_mode: Optional render mode string.
    """

    metadata: dict = {"render_modes": ["ansi"]}

    def __init__(
        self,
        tier: int = 1,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.tier = tier

        # Determine grid dimensions from tier
        self._grid_size: int = {1: 10, 2: 15, 3: 20}.get(tier, 10)

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict({
            "fov": spaces.Box(
                low=-1, high=int(max(TileType)),
                shape=(FOV_SIZE, FOV_SIZE), dtype=np.int32,
            ),
            "full_grid": spaces.Box(
                low=-1, high=int(max(TileType)),
                shape=(self._grid_size, self._grid_size), dtype=np.int32,
            ),
            "position": spaces.Box(
                low=0, high=self._grid_size - 1, shape=(2,), dtype=np.int32,
            ),
            "direction": spaces.Discrete(4),
            "health": spaces.Box(
                low=0, high=INITIAL_HP, shape=(1,), dtype=np.int32,
            ),
        })

        # Internal state — fully initialised in reset()
        self._grid: np.ndarray = np.zeros(
            (self._grid_size, self._grid_size), dtype=np.int32,
        )
        self._entities: Dict[Tuple[int, int], Entity] = {}
        self._player = PlayerState()
        self._explored: np.ndarray = np.zeros(
            (self._grid_size, self._grid_size), dtype=bool,
        )
        self._step_count: int = 0

        # Lever → barrier linkage: maps linked_id → list of (r, c)
        self._lever_barriers: Dict[str, List[Tuple[int, int]]] = {}

    # ────────────────────────────────────────────────────────
    #  Map Construction
    # ────────────────────────────────────────────────────────

    def _build_grid(self) -> None:
        """Dispatch to the tier-specific map builder."""
        self._grid[:] = TileType.EMPTY
        self._entities.clear()
        self._lever_barriers.clear()

        builders = {
            1: self._build_tier1_map,
            2: self._build_tier2_map,
            3: self._build_tier3_map,
        }
        builders.get(self.tier, self._build_tier1_map)()

    def _place_entity(self, row: int, col: int, entity: Entity) -> None:
        """Place an entity on the grid and register its metadata."""
        self._grid[row, col] = int(entity.type)
        self._entities[(row, col)] = entity

    def _fill_boundary_walls(self) -> None:
        """Fill the outermost ring of the grid with WALL tiles."""
        g = self._grid_size
        self._grid[0, :] = TileType.WALL
        self._grid[g - 1, :] = TileType.WALL
        self._grid[:, 0] = TileType.WALL
        self._grid[:, g - 1] = TileType.WALL

    def _place_barrier(
        self,
        positions: List[Tuple[int, int]],
        linked_id: str,
    ) -> None:
        """Place a WALL barrier linked to a lever/plate via linked_id."""
        self._lever_barriers[linked_id] = list(positions)
        for r, c in positions:
            self._grid[r, c] = TileType.WALL

    def _remove_barrier(self, linked_id: str) -> None:
        """Remove a wall barrier by its linked_id, replacing tiles with EMPTY."""
        positions = self._lever_barriers.pop(linked_id, [])
        for r, c in positions:
            self._grid[r, c] = TileType.EMPTY
            self._entities.pop((r, c), None)

    # ─────── Tier 1 ───────

    def _build_tier1_map(self) -> None:
        """Tier 1 — Single 10×10 room: 1 key, 1 locked door, goal chest."""
        self._fill_boundary_walls()

        self._place_entity(5, 5, Entity(
            type=TileType.DOOR, color="red", is_locked=True, item_id="key_red",
        ))
        self._place_entity(2, 2, Entity(
            type=TileType.KEY, color="red", item_id="key_red",
        ))
        self._place_entity(4, 3, Entity(
            type=TileType.HAZARD, color="orange", damage=20,
        ))
        self._place_entity(8, 8, Entity(
            type=TileType.GOAL, color="gold",
        ))

    # ─────── Tier 2 ───────

    def _build_tier2_map(self) -> None:
        """Tier 2 — 15×15 multi-room: 2 keys, 2 doors, static lava hazards."""
        self._fill_boundary_walls()

        # ── Interior partition walls ──
        # Vertical wall splitting left/right halves (col 7), gap at row 4
        for r in range(1, 14):
            if r != 4:
                self._grid[r, 7] = TileType.WALL

        # Horizontal wall splitting top/bottom in right half (row 8), gap at col 11
        for c in range(8, 14):
            if c != 11:
                self._grid[8, c] = TileType.WALL

        # ── Keys & Doors ──
        # Red key in top-left room → red door in partition
        self._place_entity(2, 3, Entity(
            type=TileType.KEY, color="red", item_id="key_red",
        ))
        self._place_entity(4, 7, Entity(
            type=TileType.DOOR, color="red", is_locked=True, item_id="key_red",
        ))

        # Blue key in bottom-right room → blue door in horizontal partition
        self._place_entity(12, 12, Entity(
            type=TileType.KEY, color="blue", item_id="key_blue",
        ))
        self._place_entity(8, 11, Entity(
            type=TileType.DOOR, color="blue", is_locked=True, item_id="key_blue",
        ))

        # ── Static lava hazards ──
        for pos in [(3, 5), (6, 3), (10, 9), (11, 10)]:
            self._place_entity(pos[0], pos[1], Entity(
                type=TileType.HAZARD, color="orange", damage=20,
            ))

        # ── Goal chest in top-right quadrant ──
        self._place_entity(3, 12, Entity(
            type=TileType.GOAL, color="gold",
        ))

    # ─────── Tier 3 ───────

    def _build_tier3_map(self) -> None:
        """Tier 3 — 20×20 hardcore dungeon with unmapped traps, levers, decoys."""
        self._fill_boundary_walls()

        # ═══════════════════════════════════════════════════
        #  Room layout (4 rooms connected by corridors)
        #
        #  Room A (top-left):     rows 1-8,   cols 1-8
        #  Room B (top-right):    rows 1-8,   cols 11-18
        #  Room C (bottom-left):  rows 11-18, cols 1-8
        #  Room D (bottom-right): rows 11-18, cols 11-18
        #
        #  Corridors at row 4 col 9-10, row 14 col 9-10
        #  Vertical corridor col 5 rows 9-10, col 14 rows 9-10
        # ═══════════════════════════════════════════════════

        # ── Vertical partition walls (col 9, col 10) ──
        for r in range(1, 19):
            self._grid[r, 9] = TileType.WALL
            self._grid[r, 10] = TileType.WALL

        # ── Horizontal partition walls (row 9, row 10) ──
        for c in range(1, 19):
            self._grid[9, c] = TileType.WALL
            self._grid[10, c] = TileType.WALL

        # ── Corridor A→B (row 4, cols 9-10) ──
        self._grid[4, 9] = TileType.EMPTY
        self._grid[4, 10] = TileType.EMPTY

        # ── Corridor C→D (row 14, cols 9-10) ──
        self._grid[14, 9] = TileType.EMPTY
        self._grid[14, 10] = TileType.EMPTY

        # ── Corridor A→C (rows 9-10, col 5) ──
        self._grid[9, 5] = TileType.EMPTY
        self._grid[10, 5] = TileType.EMPTY

        # ── Corridor B→D (rows 9-10, col 14) ──
        self._grid[9, 14] = TileType.EMPTY
        self._grid[10, 14] = TileType.EMPTY

        # ── Locked door in corridor A→B ──
        self._place_entity(4, 9, Entity(
            type=TileType.DOOR, color="red", is_locked=True, item_id="key_red",
        ))

        # ── Locked door in corridor C→D ──
        self._place_entity(14, 9, Entity(
            type=TileType.DOOR, color="blue", is_locked=True, item_id="key_blue",
        ))

        # ── Keys ──
        # Red key hidden in Room C
        self._place_entity(15, 3, Entity(
            type=TileType.KEY, color="red", item_id="key_red",
        ))
        # Blue key hidden in Room A
        self._place_entity(3, 6, Entity(
            type=TileType.KEY, color="blue", item_id="key_blue",
        ))

        # ── Lever + linked wall barrier blocking goal door ──
        # Lever in Room A removes barrier wall in Room D
        self._place_entity(2, 2, Entity(
            type=TileType.LEVER, color="green", linked_id="barrier_d",
        ))
        # Barrier blocks the corridor from Room B to Room D
        self._place_barrier([(9, 14), (10, 14)], linked_id="barrier_d")
        # Re-open corridor A→C (was overwritten by row 9/10 walls)
        # Already opened above — barrier_d blocks B→D corridor instead

        # ── Pressure plate in Room C removes inner wall segment ──
        self._place_entity(13, 3, Entity(
            type=TileType.PRESSURE_PLATE, color="gray",
            linked_id="barrier_c_inner",
        ))
        # Small inner wall in Room C
        self._place_barrier([(12, 5), (13, 5)], linked_id="barrier_c_inner")

        # ── Unmapped hidden traps (NOT in innate instincts) ──
        for pos in [(5, 3), (12, 7), (16, 15), (3, 14)]:
            self._place_entity(pos[0], pos[1], Entity(
                type=TileType.UNMAPPED_TRAP, color="invisible", damage=20,
            ))

        # ── Decoy chests (look like goals but aren't) ──
        self._place_entity(6, 6, Entity(
            type=TileType.DECOY_CHEST, color="gold",
        ))
        self._place_entity(16, 2, Entity(
            type=TileType.DECOY_CHEST, color="gold",
        ))

        # ── Static lava hazards ──
        for pos in [(7, 4), (11, 13), (17, 16)]:
            self._place_entity(pos[0], pos[1], Entity(
                type=TileType.HAZARD, color="orange", damage=20,
            ))

        # ── Real goal chest in Room D ──
        self._place_entity(16, 16, Entity(
            type=TileType.GOAL, color="gold",
        ))

    # ────────────────────────────────────────────────────────
    #  Fog-of-War & Egocentric Vision
    # ────────────────────────────────────────────────────────

    def _fov_to_world(self, fov_r: int, fov_c: int) -> Tuple[int, int]:
        """Map local FOV coords to world grid coords.

        In the egocentric frame the player sits at [4, 2]
        (bottom-centre) and row 0 is the farthest forward.
        """
        pr, pc = self._player.position
        d = self._player.direction

        if d == Direction.NORTH:
            return (pr - 4 + fov_r, pc - 2 + fov_c)
        if d == Direction.EAST:
            return (pr - 2 + fov_c, pc + 4 - fov_r)
        if d == Direction.SOUTH:
            return (pr + 4 - fov_r, pc + 2 - fov_c)
        # WEST
        return (pr + 2 - fov_c, pc - 4 + fov_r)

    def _update_explored(self) -> None:
        """Mark every tile in the current 5×5 FOV as explored."""
        g = self._grid_size
        for lr in range(FOV_SIZE):
            for lc in range(FOV_SIZE):
                wr, wc = self._fov_to_world(lr, lc)
                if 0 <= wr < g and 0 <= wc < g:
                    self._explored[wr, wc] = True

    def get_egocentric_fov(self) -> np.ndarray:
        """Compute the 5×5 forward-facing egocentric vision slice.

        Player is at [4, 2].  Tiles beyond the grid boundary
        are rendered as WALL.

        Returns:
            np.ndarray: shape (5, 5), dtype int32 of TileType values.
        """
        g = self._grid_size
        fov = np.full((FOV_SIZE, FOV_SIZE), int(TileType.WALL), dtype=np.int32)
        for lr in range(FOV_SIZE):
            for lc in range(FOV_SIZE):
                wr, wc = self._fov_to_world(lr, lc)
                if 0 <= wr < g and 0 <= wc < g:
                    fov[lr, lc] = self._grid[wr, wc]
        return fov

    # ────────────────────────────────────────────────────────
    #  Observation / Info Builders
    # ────────────────────────────────────────────────────────

    def _get_obs(self) -> Dict[str, Any]:
        """Assemble the observation dictionary."""
        g = self._grid_size
        full_grid = np.full((g, g), FOG, dtype=np.int32)
        full_grid[self._explored] = self._grid[self._explored]

        return {
            "fov": self.get_egocentric_fov(),
            "full_grid": full_grid,
            "player_state": {
                "position": self._player.position,
                "direction": self._player.direction,
                "health": self._player.health,
                "inventory": list(self._player.inventory),
            },
        }

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary diagnostic info."""
        g = self._grid_size
        return {
            "step_count": self._step_count,
            "explored_pct": float(self._explored.sum()) / (g * g),
        }

    # ────────────────────────────────────────────────────────
    #  Gymnasium API: reset / step / render
    # ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to its initial state.

        Args:
            seed:    Optional RNG seed for reproducibility.
            options: Reserved for future use.

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)

        g = self._grid_size
        self._grid = np.zeros((g, g), dtype=np.int32)
        self._explored = np.zeros((g, g), dtype=bool)

        self._build_grid()
        self._player = PlayerState(
            position=(1, 1),
            direction=Direction.EAST,
            health=INITIAL_HP,
            inventory=[],
        )
        self._step_count = 0

        # Initial FOV exploration
        self._update_explored()

        return self._get_obs(), self._get_info()

    def step(
        self, action: int,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Integer action code (see Action enum).

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        act = Action(action)
        reward: float = STEP_PENALTY
        terminated: bool = False
        truncated: bool = False
        g = self._grid_size

        self._step_count += 1

        # ── Rotations ────────────────────────────────────
        if act == Action.TURN_LEFT:
            self._player.direction = Direction((self._player.direction - 1) % 4)

        elif act == Action.TURN_RIGHT:
            self._player.direction = Direction((self._player.direction + 1) % 4)

        # ── Forward Movement ─────────────────────────────
        elif act == Action.MOVE_FORWARD:
            dr, dc = _DIR_DELTA[self._player.direction]
            nr, nc = self._player.position[0] + dr, self._player.position[1] + dc

            if not (0 <= nr < g and 0 <= nc < g):
                # Out of bounds — treat as wall
                pass
            else:
                target_tile = TileType(self._grid[nr, nc])

                can_move = True
                if target_tile == TileType.WALL:
                    can_move = False
                elif target_tile == TileType.DOOR:
                    ent = self._entities.get((nr, nc))
                    if ent is not None and ent.is_locked:
                        can_move = False  # Locked door blocks movement

                if can_move:
                    self._player.position = (nr, nc)

                    # Hazard damage
                    if target_tile == TileType.HAZARD:
                        ent = self._entities.get((nr, nc))
                        dmg = ent.damage if ent else 20
                        self._player.health = max(
                            0, self._player.health - dmg,
                        )
                        if self._player.health <= 0:
                            terminated = True

                    # Unmapped trap damage (surprise mechanic)
                    elif target_tile == TileType.UNMAPPED_TRAP:
                        ent = self._entities.get((nr, nc))
                        dmg = ent.damage if ent else 20
                        self._player.health = max(
                            0, self._player.health - dmg,
                        )
                        if self._player.health <= 0:
                            terminated = True

                    # Pressure plate activation (step-triggered)
                    elif target_tile == TileType.PRESSURE_PLATE:
                        ent = self._entities.get((nr, nc))
                        if ent and ent.linked_id:
                            self._remove_barrier(ent.linked_id)

                    # Goal reached
                    elif target_tile == TileType.GOAL:
                        reward += GOAL_REWARD
                        terminated = True

                    # Decoy chest — no reward, no termination
                    elif target_tile == TileType.DECOY_CHEST:
                        pass  # intentional no-op

        # ── Pick Up ──────────────────────────────────────
        elif act == Action.PICK_UP:
            pr, pc = self._player.position
            tile = TileType(self._grid[pr, pc])
            if tile == TileType.KEY:
                ent = self._entities.pop((pr, pc), None)
                if ent is not None:
                    self._player.inventory.append(ent.item_id)
                self._grid[pr, pc] = int(TileType.EMPTY)

        # ── Toggle / Interact ────────────────────────────
        elif act == Action.TOGGLE_INTERACT:
            dr, dc = _DIR_DELTA[self._player.direction]
            tr, tc = self._player.position[0] + dr, self._player.position[1] + dc
            if 0 <= tr < g and 0 <= tc < g:
                tile = TileType(self._grid[tr, tc])

                # Door unlock
                if tile == TileType.DOOR:
                    ent = self._entities.get((tr, tc))
                    if ent is not None and ent.is_locked:
                        if ent.item_id in self._player.inventory:
                            ent.is_locked = False

                # Lever toggle
                elif tile == TileType.LEVER:
                    ent = self._entities.get((tr, tc))
                    if ent and ent.linked_id:
                        self._remove_barrier(ent.linked_id)

        # ── Truncation check ─────────────────────────────
        if self._step_count >= MAX_STEPS:
            truncated = True

        # ── Update fog-of-war ────────────────────────────
        self._update_explored()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[str]:
        """Render the grid as an ANSI string.

        Returns:
            Multi-line string if render_mode == 'ansi', else None.
        """
        if self.render_mode != "ansi":
            return None

        g = self._grid_size

        _TILE_CHAR = {
            TileType.EMPTY:          " . ",
            TileType.WALL:           "███",
            TileType.DOOR:           " 🚪",
            TileType.KEY:            " 🔑",
            TileType.HAZARD:         " 🔥",
            TileType.GOAL:           " ⭐",
            TileType.UNMAPPED_TRAP:  " . ",   # Invisible to player
            TileType.LEVER:          " 🔧",
            TileType.PRESSURE_PLATE: " ◈ ",
            TileType.DECOY_CHEST:    " ⭐",   # Looks identical to GOAL
        }
        _DIR_CHAR = {
            Direction.NORTH: " ▲ ",
            Direction.EAST:  " ▶ ",
            Direction.SOUTH: " ▼ ",
            Direction.WEST:  " ◀ ",
        }

        lines: list[str] = []
        for r in range(g):
            row_chars: list[str] = []
            for c in range(g):
                if (r, c) == self._player.position:
                    row_chars.append(_DIR_CHAR[self._player.direction])
                elif not self._explored[r, c]:
                    row_chars.append("░░░")
                else:
                    row_chars.append(_TILE_CHAR.get(
                        TileType(self._grid[r, c]), " ? ",
                    ))
            lines.append("".join(row_chars))

        # Status bar
        p = self._player
        lines.append(
            f"HP:{p.health:3d} | Dir:{p.direction.name:5s} | "
            f"Pos:{p.position} | Inv:{p.inventory} | Step:{self._step_count}"
        )
        return "\n".join(lines)
