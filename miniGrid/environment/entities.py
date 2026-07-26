"""
entities.py — Entity Definitions for the Neuro-Symbolic RPG Grid
================================================================
Defines enumerations for directions, actions, and tile types, plus
dataclass structures for world entities and mutable player state.

Target: Python 3.10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple


class Direction(IntEnum):
    """Cardinal directions for agent orientation on the grid."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Action(IntEnum):
    """Discrete action space for the RPG environment.

    Actions:
        TURN_LEFT:        Rotate 90° counter-clockwise.
        TURN_RIGHT:       Rotate 90° clockwise.
        MOVE_FORWARD:     Advance one tile in the facing direction.
        PICK_UP:          Collect an item on the current tile.
        TOGGLE_INTERACT:  Toggle a door/lever on the adjacent forward tile.
    """

    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2
    PICK_UP = 3
    TOGGLE_INTERACT = 4


class TileType(IntEnum):
    """Tile classification codes used in the grid map.

    Values:
        EMPTY:  Passable open floor.
        WALL:   Impassable boundary or obstacle.
        DOOR:   May be locked; requires matching key to toggle.
        KEY:    Collectible item that unlocks a corresponding door.
        HAZARD: Deals damage on contact (e.g. lava, spikes).
        GOAL:   Level objective; reaching it ends the episode.
    """

    EMPTY = 0
    WALL = 1
    DOOR = 2
    KEY = 3
    HAZARD = 4
    GOAL = 5


# Sentinel value for unexplored / fog-of-war tiles in observations.
FOG: int = -1


@dataclass
class Entity:
    """Metadata for a placed world entity occupying a grid tile.

    Attributes:
        type:      The TileType classification of this entity.
        color:     Colour identifier used for key/door matching and rendering.
        is_locked: Whether this entity blocks passage (doors only).
        item_id:   Unique string linking keys to their matching doors.
        damage:    HP damage inflicted when the player steps on this tile.
    """

    type: TileType
    color: str = "white"
    is_locked: bool = False
    item_id: str = ""
    damage: int = 0


@dataclass
class PlayerState:
    """Mutable agent state container.

    Attributes:
        position:  (row, col) coordinate on the 10×10 grid.
        direction: Cardinal direction the agent is currently facing.
        health:    Hit-points; game over when this reaches zero.
        inventory: List of collected item_id strings.
    """

    position: Tuple[int, int] = (1, 1)
    direction: Direction = Direction.EAST
    health: int = 100
    inventory: List[str] = field(default_factory=list)
