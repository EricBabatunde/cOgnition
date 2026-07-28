"""
symbolic_engine.py — Symbolic Logic Engine (Subsystem B)
=========================================================
Implements formal safety verification for proposed agent actions
using the Z3 SMT solver.  Supports both static hard-coded axioms
and dynamically loaded JSON-based rules with confidence-bound
filtering and Z3 SMT translation.

Safety Axioms (hard-coded baseline):
  1. **Solid Obstacle Invariant** — Cannot move into walls.
  2. **Hazard Protection Invariant** — Cannot move into hazards.
  3. **Locked Door Access Invariant** — Cannot move through a
     locked door without the matching key.
  4. **Interaction Precondition** — Cannot toggle a locked door
     without the matching key.

Dynamic rules extend this baseline via ``load_rules_from_config``
and ``verify_action_dynamic``.

Target: Python 3.10
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import z3


# ──────────────────────────────────────────────
# Rule dataclass
# ──────────────────────────────────────────────

@dataclass
class SymbolicRule:
    """A single domain rule parsed from JSON configuration.

    Attributes:
        rule_id:    Unique identifier (e.g. ``"wall_blocking"``).
        premise:    Predicate string (e.g. ``"InFront(Wall)"``).
        conclusion: Outcome string (e.g. ``"Forbid(MOVE_FORWARD)"``).
        confidence: Trust weight in [0.0, 1.0].
        cached_expr: Pre-compiled Z3 expression for fast evaluation.
    """

    rule_id: str
    premise: str
    conclusion: str
    confidence: float = 1.0
    cached_expr: Optional[z3.ExprRef] = field(default=None, repr=False)


# ──────────────────────────────────────────────
# Predicate mapping helpers
# ──────────────────────────────────────────────

def _parse_premise(premise: str, z3v: Dict[str, z3.ExprRef]) -> Optional[z3.ExprRef]:
    """Translate a premise string to a Z3 expression.

    Supported premise patterns:
      - ``InFront(Wall)``
      - ``InFront(Hazard)``
      - ``InFront(Door) AND IsLocked(Door)``
      - ``SeeInFOV(GoalChest)``

    Args:
        premise: Raw premise string from the rule.
        z3v:     Dictionary of named Z3 Boolean variables.

    Returns:
        Z3 expression, or ``None`` if the premise is unrecognised.
    """
    p = premise.strip()

    if p == "InFront(Wall)":
        return z3v["is_wall"]

    if p == "InFront(Hazard)":
        return z3v["is_hazard"]

    if p == "InFront(Door) AND IsLocked(Door)":
        return z3.And(z3v["is_door"], z3v["is_locked"])

    if p == "InFront(Door) AND HasItem(MatchingKey)":
        return z3.And(z3v["is_door"], z3v["has_key"])

    if p == "InFront(Door)":
        return z3v["is_door"]

    if p == "SeeInFOV(GoalChest)":
        return z3v.get("see_goal", z3.BoolVal(True))

    return None


def _parse_conclusion(
    conclusion: str,
    z3v: Dict[str, z3.ExprRef],
) -> Optional[z3.ExprRef]:
    """Translate a conclusion string to a Z3 constraint.

    Supported conclusion patterns:
      - ``Forbid(MOVE_FORWARD)``  → act_move implies False
      - ``RequiresItem(MatchingKey)`` → act_move implies has_key
      - ``SetPriorityTarget(GoalChest)`` → tautology (always SAT)

    Args:
        conclusion: Raw conclusion string from the rule.
        z3v:        Dictionary of named Z3 Boolean variables.

    Returns:
        Z3 expression, or ``None`` if unrecognised.
    """
    c = conclusion.strip()

    if c == "Forbid(MOVE_FORWARD)":
        return z3.Implies(z3v["act_move"], z3.BoolVal(False))

    if c == "RequiresItem(MatchingKey)":
        return z3.Implies(z3v["act_move"], z3v["has_key"])

    if c == "AllowAction(TOGGLE_INTERACT)":
        return z3.Implies(z3v["act_interact"], z3.BoolVal(True))

    if c == "SetPriorityTarget(GoalChest)":
        # Advisory rule — does not constrain actions
        return z3.BoolVal(True)

    return None


# ──────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────

class SymbolicLogicEngine:
    """First-order safety verification engine backed by Z3.

    Supports both static axioms (``verify_action_safety``) and
    dynamically loaded JSON rules (``verify_action_dynamic``) with
    confidence-threshold filtering.

    Attributes:
        axiom_ids:               Static axiom identifiers.
        loaded_rules:            Dynamically loaded ``SymbolicRule`` list.
        min_confidence_threshold: Default confidence filter for dynamic rules.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.80,
    ) -> None:
        """Initialise the symbolic engine.

        Args:
            min_confidence_threshold: Minimum confidence for a
                dynamic rule to be asserted during verification.
        """
        # Static axiom identifiers (for diagnostics)
        self.axiom_ids: List[str] = [
            "solid_obstacle_invariant",
            "hazard_protection_invariant",
            "locked_door_access_invariant",
            "interaction_precondition",
        ]

        # Dynamic rule store
        self.loaded_rules: List[SymbolicRule] = []
        self.min_confidence_threshold: float = min_confidence_threshold

        # ── Pre-allocated Z3 Boolean variables ──
        self.z3_vars: Dict[str, z3.ExprRef] = {
            "is_wall": z3.Bool("is_wall"),
            "is_hazard": z3.Bool("is_hazard"),
            "is_door": z3.Bool("is_door"),
            "is_locked": z3.Bool("is_locked"),
            "has_key": z3.Bool("has_key"),
            "act_move": z3.Bool("act_move"),
            "act_interact": z3.Bool("act_interact"),
        }

        # ── Persistent Z3 Solver ──
        self.solver = z3.Solver()

    # ────────────────────────────────────────────
    #  Dynamic rule loading
    # ────────────────────────────────────────────

    def load_rules_from_config(
        self,
        config_path: str = "config/innate_instincts.json",
    ) -> int:
        """Load domain rules from a JSON configuration file.

        Parses the ``innate_rules`` array and appends each entry
        as a ``SymbolicRule`` to ``self.loaded_rules``.

        Args:
            config_path: Path to the JSON config.  Relative paths
                         are resolved from the project root.

        Returns:
            Number of rules successfully loaded.

        Raises:
            FileNotFoundError: If the config file is missing.
            json.JSONDecodeError: If the JSON is malformed.
        """
        if not os.path.isabs(config_path):
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)),
            )
            config_path = os.path.join(base_dir, config_path)

        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        count = 0
        for entry in data.get("innate_rules", []):
            try:
                rule = SymbolicRule(
                    rule_id=entry["rule_id"],
                    premise=entry["premise"],
                    conclusion=entry["conclusion"],
                    confidence=float(entry.get("confidence", 1.0)),
                )
                # Pre-compile to Z3 AST
                rule.cached_expr = self.parse_rule_to_z3(rule, self.z3_vars)
                self.loaded_rules.append(rule)
                count += 1
            except (KeyError, TypeError, ValueError):
                # Skip malformed entries gracefully
                continue

        return count

    # ────────────────────────────────────────────
    #  Rule → Z3 translator
    # ────────────────────────────────────────────

    def parse_rule_to_z3(
        self,
        rule: SymbolicRule,
        z3_vars: Dict[str, z3.ExprRef],
    ) -> Optional[z3.ExprRef]:
        """Translate a ``SymbolicRule`` into a Z3 proposition.

        Combines the parsed premise and conclusion into a single
        ``z3.Implies(premise, conclusion)`` constraint.

        Args:
            rule:     The symbolic rule to translate.
            z3_vars:  Dictionary of named Z3 Boolean variables.

        Returns:
            A Z3 expression, or ``None`` if either the premise or
            conclusion is unrecognised.
        """
        premise_expr = _parse_premise(rule.premise, z3_vars)
        conclusion_expr = _parse_conclusion(rule.conclusion, z3_vars)

        if premise_expr is None or conclusion_expr is None:
            return None

        return z3.Implies(premise_expr, conclusion_expr)

    # ────────────────────────────────────────────
    #  Dynamic verification
    # ────────────────────────────────────────────

    def verify_action_dynamic(
        self,
        action_name: str,
        state_context: Dict[str, Any],
        min_confidence: Optional[float] = None,
    ) -> Tuple[bool, str, str, List[str]]:
        """Verify an action against dynamically loaded rules.

        Filters loaded rules by confidence threshold, translates
        each into a Z3 constraint, and checks satisfiability.

        Args:
            action_name:   Action label (e.g. ``"MOVE_FORWARD"``).
            state_context: Dictionary of grounding facts:
                           ``is_wall``, ``is_hazard``, ``is_door``,
                           ``is_locked``, ``has_key`` (all booleans).
            min_confidence: Override for the confidence threshold.
                            Defaults to ``self.min_confidence_threshold``.

        Returns:
            A 4-tuple ``(is_safe, explanation, status, active_rule_ids)``:
              - **is_safe** (bool): ``True`` if action is permissible.
              - **explanation** (str): Human-readable reason.
              - **status** (str): ``"SAT"`` or ``"UNSAT"``.
              - **active_rule_ids** (List[str]): IDs of rules that
                met the confidence threshold.
        """
        threshold = (
            min_confidence
            if min_confidence is not None
            else self.min_confidence_threshold
        )

        self.solver.push()
        try:
            # ── Ground state facts ──
            self.solver.add(self.z3_vars["is_wall"] == state_context.get("is_wall", False))
            self.solver.add(self.z3_vars["is_hazard"] == state_context.get("is_hazard", False))
            self.solver.add(self.z3_vars["is_door"] == state_context.get("is_door", False))
            self.solver.add(self.z3_vars["is_locked"] == state_context.get("is_locked", False))
            self.solver.add(self.z3_vars["has_key"] == state_context.get("has_key", False))

            # ── Ground proposed action ──
            action_upper = action_name.upper()
            self.solver.add(self.z3_vars["act_move"] == (action_upper == "MOVE_FORWARD"))
            self.solver.add(self.z3_vars["act_interact"] == (action_upper == "TOGGLE_INTERACT"))

            # ── Filter and assert dynamic rules ──
            active_rule_ids: List[str] = []
            for rule in self.loaded_rules:
                if rule.confidence < threshold:
                    continue

                if rule.cached_expr is not None:
                    self.solver.add(rule.cached_expr)
                    active_rule_ids.append(rule.rule_id)

            # ── Solve ──
            status = self.solver.check()

            if status == z3.sat:
                return (
                    True,
                    "Action mathematically verified safe",
                    "SAT",
                    active_rule_ids,
                )

            return (
                False,
                f"Safety violation detected for action {action_name} "
                f"(rules: {', '.join(active_rule_ids)})",
                "UNSAT",
                active_rule_ids,
            )
        finally:
            self.solver.pop()

    # ────────────────────────────────────────────
    #  Static verification (backwards compatible)
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
        current world state and the four hard-coded safety axioms,
        then checks whether the proposed action is consistent with
        all constraints.

        This method is preserved for backwards compatibility.
        See ``verify_action_dynamic`` for the rule-driven pipeline.

        Args:
            action_name:      Action label (e.g. ``"MOVE_FORWARD"``).
            target_tile_type: Tile classification of the forward cell.
            is_locked:        Whether the target door is locked.
            inventory:        List of item-ID strings the agent holds.

        Returns:
            A 3-tuple ``(safe, reason, status)``.
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

        # ── Ground state facts ──
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
        solver.add(z3.Implies(var_act_move, z3.Not(var_is_wall)))
        solver.add(z3.Implies(var_act_move, z3.Not(var_is_hazard)))
        solver.add(
            z3.Implies(
                z3.And(var_act_move, var_is_door, var_is_locked),
                var_has_key,
            )
        )
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

        return (
            False,
            f"Safety violation detected for action {action_name} "
            f"on {target_tile_type}",
            "UNSAT",
        )
