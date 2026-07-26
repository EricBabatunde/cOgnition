"""
custom_rpg_env.py — Gymnasium-Compatible RPG Grid Environment
=============================================================
Implements a 10×10 tile-based environment with fog-of-war, item
mechanics, hazard damage, and goal-seeking dynamics for the
Neuro-Symbolic Cognitive Engine.

Target: Python 3.10
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

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

GRID_SIZE: int = 10
FOV_SIZE: int = 5
MAX_STEPS: int = 200
INITIAL_HP: int = 100
STEP_PENALTY: float = -0.1
GOAL_REWARD: float = 100.0


class CustomRPGEnv(gymnasium.Env):
    """A 10×10 tile-based RPG environment with fog-of-war.

    Observation dict keys:
        fov          — (5, 5) int32 egocentric vision slice.
        full_grid    — (10, 10) int32 world map (FOG = -1 for unexplored).
        player_state — dict with position, direction, health, inventory.

    Reward mechanics:
        Every step   →  -0.1  (step penalty)
        Hazard tile  →  HP reduced by entity damage
        Goal reached →  +100, episode terminates
        HP ≤ 0       →  episode terminates (reward = 0)
    """

    metadata: dict = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict({
            "fov": spaces.Box(
                low=-1, high=int(max(TileType)),
                shape=(FOV_SIZE, FOV_SIZE), dtype=np.int32,
            ),
            "full_grid": spaces.Box(
                low=-1, high=int(max(TileType)),
                shape=(GRID_SIZE, GRID_SIZE), dtype=np.int32,
            ),
            "position": spaces.Box(
                low=0, high=GRID_SIZE - 1, shape=(2,), dtype=np.int32,
            ),
            "direction": spaces.Discrete(4),
            "health": spaces.Box(low=0, high=INITIAL_HP, shape=(1,), dtype=np.int32),
        })

        # Internal state — fully initialised in reset()
        self._grid: np.ndarray = np.zeros(
            (GRID_SIZE, GRID_SIZE), dtype=np.int32,
        )
        self._entities: Dict[Tuple[int, int], Entity] = {}
        self._player = PlayerState()
        self._explored: np.ndarray = np.zeros(
            (GRID_SIZE, GRID_SIZE), dtype=bool,
        )
        self._step_count: int = 0

    # ────────────────────────────────────────────────────────
    #  Map Construction
    # ────────────────────────────────────────────────────────

    def _build_grid(self) -> None:
        """Construct the 10×10 grid with boundary walls and starter entities."""
        self._grid[:] = TileType.EMPTY
        self._entities.clear()

        # Boundary walls (rows 0/9, cols 0/9)
        self._grid[0, :] = TileType.WALL
        self._grid[-1, :] = TileType.WALL
        self._grid[:, 0] = TileType.WALL
        self._grid[:, -1] = TileType.WALL

        # Starter room layout
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

    def _place_entity(self, row: int, col: int, entity: Entity) -> None:
        """Place an entity on the grid and register its metadata."""
        self._grid[row, col] = int(entity.type)
        self._entities[(row, col)] = entity

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
        for lr in range(FOV_SIZE):
            for lc in range(FOV_SIZE):
                wr, wc = self._fov_to_world(lr, lc)
                if 0 <= wr < GRID_SIZE and 0 <= wc < GRID_SIZE:
                    self._explored[wr, wc] = True

    def get_egocentric_fov(self) -> np.ndarray:
        """Compute the 5×5 forward-facing egocentric vision slice.

        Player is at [4, 2].  Tiles beyond the grid boundary
        are rendered as WALL.

        Returns:
            np.ndarray: shape (5, 5), dtype int32 of TileType values.
        """
        fov = np.full((FOV_SIZE, FOV_SIZE), int(TileType.WALL), dtype=np.int32)
        for lr in range(FOV_SIZE):
            for lc in range(FOV_SIZE):
                wr, wc = self._fov_to_world(lr, lc)
                if 0 <= wr < GRID_SIZE and 0 <= wc < GRID_SIZE:
                    fov[lr, lc] = self._grid[wr, wc]
        return fov

    # ────────────────────────────────────────────────────────
    #  Observation / Info Builders
    # ────────────────────────────────────────────────────────

    def _get_obs(self) -> Dict[str, Any]:
        """Assemble the observation dictionary."""
        full_grid = np.full((GRID_SIZE, GRID_SIZE), FOG, dtype=np.int32)
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
        return {
            "step_count": self._step_count,
            "explored_pct": float(self._explored.sum()) / (GRID_SIZE * GRID_SIZE),
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

        self._build_grid()
        self._player = PlayerState(
            position=(1, 1),
            direction=Direction.EAST,
            health=INITIAL_HP,
            inventory=[],
        )
        self._explored = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
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
                    self._player.health = max(0, self._player.health - dmg)
                    if self._player.health <= 0:
                        terminated = True

                # Goal reached
                if target_tile == TileType.GOAL:
                    reward += GOAL_REWARD
                    terminated = True

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
            if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                tile = TileType(self._grid[tr, tc])
                if tile == TileType.DOOR:
                    ent = self._entities.get((tr, tc))
                    if ent is not None and ent.is_locked:
                        if ent.item_id in self._player.inventory:
                            ent.is_locked = False

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

        _TILE_CHAR = {
            TileType.EMPTY:  " . ",
            TileType.WALL:   "███",
            TileType.DOOR:   " 🚪",
            TileType.KEY:    " 🔑",
            TileType.HAZARD: " 🔥",
            TileType.GOAL:   " ⭐",
        }
        _DIR_CHAR = {
            Direction.NORTH: " ▲ ",
            Direction.EAST:  " ▶ ",
            Direction.SOUTH: " ▼ ",
            Direction.WEST:  " ◀ ",
        }

        lines: list[str] = []
        for r in range(GRID_SIZE):
            row_chars: list[str] = []
            for c in range(GRID_SIZE):
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
