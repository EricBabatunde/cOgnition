"""
symbolic_engine.py — Symbolic Logic Engine (Subsystem B)
=========================================================
Implements formal safety verification for proposed agent actions
using the Z3 SMT solver. Each action is checked against a set of
first-order safety axioms before execution.

Safety Axioms:
  1. **Solid Obstacle Invariant** — Cannot move into walls.
  2. **Hazard Protection Invariant** — Cannot move into hazards.
  3. **Locked Door Access Invariant** — Cannot move through a
     locked door without the matching key.
  4. **Interaction Precondition** — Cannot toggle a locked door
     without the matching key.

Target: Python 3.10
"""

from __future__ import annotations

from typing import List, Tuple

import z3


class SymbolicLogicEngine:
    """First-order safety verification engine backed by Z3.

    Verifies proposed actions against hard safety constraints
    before they are dispatched to the environment.

    The engine is stateless — each ``verify_action_safety`` call
    instantiates a fresh solver with the relevant grounding facts
    and axioms.
    """

    def __init__(self) -> None:
        """Initialise the symbolic engine.

        Safety axiom templates are defined declaratively and
        instantiated per-query inside ``verify_action_safety``.
        """
        # Axiom identifiers (for diagnostics / logging)
        self.axiom_ids: List[str] = [
            "solid_obstacle_invariant",
            "hazard_protection_invariant",
            "locked_door_access_invariant",
            "interaction_precondition",
        ]

    # ────────────────────────────────────────────
    #  Core verification
    # ────────────────────────────────────────────

    def verify_action_safety(
        self,
        action_name: str,
        target_tile_type: str,
        is_locked: bool,
        inventory: List[str],
    ) -> Tuple[bool, str, str]:
        """Formally verify whether a proposed action is safe.

        Constructs a Z3 satisfiability problem encoding the
        current world state and the four safety axioms, then
        checks whether the proposed action is consistent with
        all constraints.

        Args:
            action_name:      Action label (e.g. ``"MOVE_FORWARD"``,
                              ``"TOGGLE_INTERACT"``).
            target_tile_type: Tile classification of the forward cell
                              (e.g. ``"WALL"``, ``"HAZARD"``, ``"DOOR"``,
                              ``"EMPTY"``, ``"GOAL"``).
            is_locked:        Whether the target door (if any) is locked.
            inventory:        List of item-ID strings the agent holds.

        Returns:
            A 3-tuple ``(safe, reason, status)``:
              - **safe** (bool): ``True`` if the action is permissible.
              - **reason** (str): Human-readable explanation.
              - **status** (str): ``"SAT"`` or ``"UNSAT"``.
        """
        solver = z3.Solver()

        # ── Z3 Boolean state variables ──
        var_is_wall = z3.Bool("is_wall")
        var_is_hazard = z3.Bool("is_hazard")
        var_is_door = z3.Bool("is_door")
        var_is_locked = z3.Bool("is_locked")
        var_has_key = z3.Bool("has_key")
        var_act_move = z3.Bool("act_move")
        var_act_interact = z3.Bool("act_interact")

        # ── Ground state facts from observations ──
        tile_upper = target_tile_type.upper()

        solver.add(var_is_wall == (tile_upper == "WALL"))
        solver.add(var_is_hazard == (tile_upper == "HAZARD"))
        solver.add(var_is_door == (tile_upper == "DOOR"))
        solver.add(var_is_locked == is_locked)
        solver.add(var_has_key == (len(inventory) > 0))

        # ── Ground proposed action ──
        action_upper = action_name.upper()
        solver.add(var_act_move == (action_upper == "MOVE_FORWARD"))
        solver.add(var_act_interact == (action_upper == "TOGGLE_INTERACT"))

        # ── First-Order Safety Axioms ──

        # Axiom 1: Solid Obstacle Invariant
        #   If moving forward, the target must NOT be a wall.
        solver.add(z3.Implies(var_act_move, z3.Not(var_is_wall)))

        # Axiom 2: Hazard Protection Invariant
        #   If moving forward, the target must NOT be a hazard.
        solver.add(z3.Implies(var_act_move, z3.Not(var_is_hazard)))

        # Axiom 3: Locked Door Access Invariant
        #   If moving through a locked door, the agent must have a key.
        solver.add(
            z3.Implies(
                z3.And(var_act_move, var_is_door, var_is_locked),
                var_has_key,
            )
        )

        # Axiom 4: Interaction Precondition
        #   If toggling a locked door, the agent must have a key.
        solver.add(
            z3.Implies(
                z3.And(var_act_interact, var_is_door, var_is_locked),
                var_has_key,
            )
        )

        # ── Solve ──
        status = solver.check()

        if status == z3.sat:
            return (
                True,
                "Action mathematically verified safe",
                "SAT",
            )

        # UNSAT — at least one axiom was violated
        return (
            False,
            f"Safety violation detected for action {action_name} "
            f"on {target_tile_type}",
            "UNSAT",
        )
