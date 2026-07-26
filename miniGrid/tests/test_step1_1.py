#!/usr/bin/env python3
"""
test_step1_1.py — Unit Verification for Phase 1.1 Environment Engine
=====================================================================
Rigorously validates CustomRPGEnv state logic, FOV grounding,
hazard/item/door mechanics against the architecture specification.

Target: Python 3.10  |  Zero external test-framework dependencies.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Tuple

# ──────────────────────────────────────────────
# ANSI formatting
# ──────────────────────────────────────────────
_G = "\033[92m"   # green
_R = "\033[91m"   # red
_C = "\033[96m"   # cyan
_B = "\033[1m"    # bold
_D = "\033[2m"    # dim
_X = "\033[0m"    # reset

PASS = f"{_G}✓ PASS{_X}"
FAIL = f"{_R}✗ FAIL{_X}"

# ──────────────────────────────────────────────
# Imports under test
# ──────────────────────────────────────────────
from environment import CustomRPGEnv, Direction, Action, TileType


# ──────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────

def test_reset_logic() -> None:
    """Test 1 — Reset Logic: initial player state and observation shape."""
    env = CustomRPGEnv()
    obs, info = env.reset(seed=0)

    ps = obs["player_state"]
    assert ps["position"] == (1, 1), \
        f"Expected position (1,1), got {ps['position']}"
    assert ps["direction"] == Direction.EAST, \
        f"Expected direction EAST, got {ps['direction']}"
    assert ps["health"] == 100, \
        f"Expected HP 100, got {ps['health']}"
    assert ps["inventory"] == [], \
        f"Expected empty inventory, got {ps['inventory']}"

    fov = obs["fov"]
    assert fov.shape == (5, 5), \
        f"Expected FOV shape (5,5), got {fov.shape}"

    full = obs["full_grid"]
    assert full.shape == (10, 10), \
        f"Expected full_grid shape (10,10), got {full.shape}"

    # Verify fog: unexplored tiles should be -1
    assert (full == -1).any(), \
        "Expected some fog (-1) tiles in initial full_grid"

    # Verify info dict
    assert "step_count" in info, "info missing 'step_count'"
    assert info["step_count"] == 0, \
        f"Expected step_count 0, got {info['step_count']}"


def test_rotation_and_wall_collision() -> None:
    """Test 2 — Rotation & Wall Collision: turns and boundary enforcement."""
    env = CustomRPGEnv()
    env.reset(seed=0)

    # ── Rotation sequence ──
    # Start: EAST
    assert env._player.direction == Direction.EAST

    # TURN_LEFT → NORTH
    env.step(Action.TURN_LEFT)
    assert env._player.direction == Direction.NORTH, \
        f"After TURN_LEFT from EAST, expected NORTH, got {env._player.direction.name}"

    # TURN_LEFT → WEST
    env.step(Action.TURN_LEFT)
    assert env._player.direction == Direction.WEST, \
        f"After TURN_LEFT from NORTH, expected WEST, got {env._player.direction.name}"

    # TURN_RIGHT → NORTH
    env.step(Action.TURN_RIGHT)
    assert env._player.direction == Direction.NORTH, \
        f"After TURN_RIGHT from WEST, expected NORTH, got {env._player.direction.name}"

    # Full CW cycle: NORTH → EAST → SOUTH → WEST → NORTH
    for expected in [Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH]:
        env.step(Action.TURN_RIGHT)
        assert env._player.direction == expected, \
            f"CW cycle: expected {expected.name}, got {env._player.direction.name}"

    # ── Wall collision ──
    # Reset fresh: player at (1,1) facing EAST
    env.reset(seed=0)

    # Face NORTH (toward wall at row 0)
    env.step(Action.TURN_LEFT)  # EAST → NORTH
    assert env._player.direction == Direction.NORTH

    # Try to walk into wall at (0,1)
    env.step(Action.MOVE_FORWARD)
    assert env._player.position == (1, 1), \
        f"Should be blocked by north wall, but moved to {env._player.position}"

    # Face WEST (toward wall at col 0)
    env.step(Action.TURN_LEFT)  # NORTH → WEST
    env.step(Action.MOVE_FORWARD)
    assert env._player.position == (1, 1), \
        f"Should be blocked by west wall, but moved to {env._player.position}"


def test_egocentric_fov_grounding() -> None:
    """Test 3 — Egocentric FOV Grounding: rotation-dependent vision slices."""
    env = CustomRPGEnv()
    env.reset(seed=0)

    # Place a sentinel entity at (1, 3) — 2 tiles east of player at (1,1)
    env._grid[1, 3] = int(TileType.HAZARD)

    # ── Facing EAST: sentinel should be at FOV[4, 4] (right of player) ──
    # Player at [4,2] in FOV.  (1,3) is 2 cols east = 2 forward when EAST.
    # FOV mapping EAST: world_col = pc + 4 - fov_r → fov_r = pc+4-wc = 1+4-3 = 2
    #                   world_row = pr - 2 + fov_c → fov_c = wr-pr+2 = 1-1+2 = 2
    # So sentinel is at FOV[2, 2] when facing EAST
    fov_east = env.get_egocentric_fov()
    assert fov_east[2, 2] == TileType.HAZARD, \
        f"EAST FOV: expected HAZARD at [2,2], got {TileType(fov_east[2, 2]).name}"

    # ── Facing NORTH: same world tile (1,3) should shift in the FOV ──
    env.step(Action.TURN_LEFT)  # → NORTH
    # FOV mapping NORTH: world_row = pr-4+fov_r → fov_r = wr-pr+4 = 1-1+4 = 4
    #                    world_col = pc-2+fov_c → fov_c = wc-pc+2 = 3-1+2 = 4
    # Sentinel at FOV[4, 4] when facing NORTH
    fov_north = env.get_egocentric_fov()
    assert fov_north[4, 4] == TileType.HAZARD, \
        f"NORTH FOV: expected HAZARD at [4,4], got {TileType(fov_north[4, 4]).name}"

    # ── Verify player self-position: always at FOV[4, 2] ──
    # (the tile under the player — should be EMPTY since player started at (1,1))
    for _ in range(4):
        fov = env.get_egocentric_fov()
        assert fov[4, 2] == TileType.EMPTY, \
            f"Player tile FOV[4,2] should be EMPTY, got {TileType(fov[4, 2]).name}"
        env.step(Action.TURN_RIGHT)

    # Clean up sentinel
    env._grid[1, 3] = int(TileType.EMPTY)


def test_hazard_damage() -> None:
    """Test 4 — Hazard Damage: stepping onto hazard at (4,3) reduces HP."""
    env = CustomRPGEnv()
    env.reset(seed=0)

    # Teleport player adjacent to hazard at (4,3), facing SOUTH
    env._player.position = (3, 3)
    env._player.direction = Direction.SOUTH
    assert env._player.health == 100

    # Step onto hazard
    obs, reward, terminated, truncated, info = env.step(Action.MOVE_FORWARD)
    ps = obs["player_state"]

    assert ps["position"] == (4, 3), \
        f"Expected position (4,3), got {ps['position']}"
    assert ps["health"] == 80, \
        f"Expected HP 80 after 20-damage hazard, got {ps['health']}"
    assert not terminated, \
        "Agent should survive with 80 HP"

    # ── Lethal accumulation: step on hazard 4 more times → HP 0 ──
    for _ in range(3):
        env._player.position = (3, 3)  # reset above hazard
        env.step(Action.MOVE_FORWARD)

    assert env._player.health == 20, \
        f"Expected HP 20 after 4 hazard hits total, got {env._player.health}"

    # Final lethal hit
    env._player.position = (3, 3)
    _, _, terminated, _, _ = env.step(Action.MOVE_FORWARD)
    assert env._player.health == 0, \
        f"Expected HP 0, got {env._player.health}"
    assert terminated, \
        "Episode should terminate when HP reaches 0"


def test_item_pickup() -> None:
    """Test 5 — Item Pickup: collecting the key at (2,2)."""
    env = CustomRPGEnv()
    env.reset(seed=0)

    # Teleport player onto the key tile
    env._player.position = (2, 2)
    assert env._grid[2, 2] == TileType.KEY, \
        f"Expected KEY at (2,2), got {TileType(env._grid[2, 2]).name}"

    # Pick up
    obs, _, _, _, _ = env.step(Action.PICK_UP)
    ps = obs["player_state"]

    assert "key_red" in ps["inventory"], \
        f"Expected 'key_red' in inventory, got {ps['inventory']}"
    assert env._grid[2, 2] == TileType.EMPTY, \
        f"Key tile should become EMPTY after pickup, got {TileType(env._grid[2, 2]).name}"
    assert (2, 2) not in env._entities, \
        "Entity metadata should be removed after pickup"

    # Picking up again on an empty tile should be a no-op
    obs2, _, _, _, _ = env.step(Action.PICK_UP)
    assert obs2["player_state"]["inventory"] == ["key_red"], \
        "Inventory should remain unchanged after picking up on empty tile"


def test_door_interaction() -> None:
    """Test 6 — Door Interaction: locked/unlocked state transitions."""
    env = CustomRPGEnv()
    env.reset(seed=0)

    # ── Phase A: interact WITHOUT key → door stays locked ──
    # Teleport to (5,4) facing EAST, door at (5,5)
    env._player.position = (5, 4)
    env._player.direction = Direction.EAST
    env._player.inventory = []

    door_ent = env._entities.get((5, 5))
    assert door_ent is not None, "Door entity missing at (5,5)"
    assert door_ent.is_locked, "Door should start locked"

    env.step(Action.TOGGLE_INTERACT)
    assert door_ent.is_locked, \
        "Door should remain locked without matching key"

    # Try to walk through locked door
    env.step(Action.MOVE_FORWARD)
    assert env._player.position == (5, 4), \
        f"Should be blocked by locked door, but moved to {env._player.position}"

    # ── Phase B: interact WITH key → door unlocks ──
    env._player.inventory = ["key_red"]

    env.step(Action.TOGGLE_INTERACT)
    assert not door_ent.is_locked, \
        "Door should be unlocked after TOGGLE_INTERACT with matching key"

    # Walk through unlocked door
    env.step(Action.MOVE_FORWARD)
    assert env._player.position == (5, 5), \
        f"Should walk through unlocked door to (5,5), but at {env._player.position}"


# ──────────────────────────────────────────────
# Test Runner
# ──────────────────────────────────────────────

_TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("Test 1 — Reset Logic",                test_reset_logic),
    ("Test 2 — Rotation & Wall Collision",   test_rotation_and_wall_collision),
    ("Test 3 — Egocentric FOV Grounding",    test_egocentric_fov_grounding),
    ("Test 4 — Hazard Damage",               test_hazard_damage),
    ("Test 5 — Item Pickup",                 test_item_pickup),
    ("Test 6 — Door Interaction",            test_door_interaction),
]


def main() -> int:
    """Execute all test cases and print a formatted summary."""
    width = 62

    print(f"\n{_B}{'═' * width}{_X}")
    print(f"{_B}  PHASE 1.1 — ENVIRONMENT ENGINE UNIT VERIFICATION{_X}")
    print(f"{_B}{'═' * width}{_X}")

    results: List[Tuple[str, bool, str]] = []

    for name, fn in _TESTS:
        print(f"\n{_C}{'━' * width}{_X}")
        print(f"{_C}┃{_X} {_B}{name}{_X}")
        print(f"{_C}{'━' * width}{_X}")

        try:
            fn()
            results.append((name, True, ""))
            print(f"  {_D}└─{_X} [{PASS}] {name}")
        except AssertionError as exc:
            results.append((name, False, str(exc)))
            print(f"  {_D}└─{_X} [{FAIL}] {name}")
            print(f"       {_R}{exc}{_X}")
            traceback.print_exc()
        except Exception as exc:
            results.append((name, False, str(exc)))
            print(f"  {_D}└─{_X} [{FAIL}] {name}")
            print(f"       {_R}Unexpected: {exc}{_X}")
            traceback.print_exc()

    # ── Summary ──
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{_B}{'═' * width}{_X}")
    print(f"  {_B}TEST SUMMARY{_X}")
    print(f"{'─' * width}")
    for name, ok, err in results:
        tag = PASS if ok else FAIL
        suffix = f"  {_D}({err}){_X}" if err else ""
        print(f"  [{tag}]  {name}{suffix}")
    print(f"{'─' * width}")

    if failed == 0:
        print(f"  {_G}{_B}{passed}/{total} TESTS PASSED — Environment engine verified.{_X}")
    else:
        print(f"  {_R}{_B}{failed}/{total} TESTS FAILED — Review errors above.{_X}")

    print(f"{_B}{'═' * width}{_X}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
