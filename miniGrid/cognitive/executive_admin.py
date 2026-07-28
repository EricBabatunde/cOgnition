"""
executive_admin.py — Subsystem D Metacognitive Goal Engine
==========================================================
Implements high-level goal synthesis and sub-goal decomposition.
Evaluates the core knowledge graph to formulate action plans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Any, Dict

from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine


class GoalType(Enum):
    """Types of high-level goals the agent can pursue."""
    REACH_EXIT = "REACH_EXIT"
    UNLOCK_DOOR = "UNLOCK_DOOR"
    FETCH_KEY = "FETCH_KEY"
    EXPLORE_FRONTIER = "EXPLORE_FRONTIER"


@dataclass
class SubGoal:
    """Represents a discrete step in a broader goal plan."""
    goal_type: GoalType
    target_pos: Tuple[int, int]
    target_entity: Optional[str] = None
    is_completed: bool = False


class ExecutiveGoalEngine:
    """Synthesizes high-level goals and decomposes them into executable plans."""

    def __init__(self, matrix: CoreKnowledgeMatrix, logic_engine: SymbolicLogicEngine) -> None:
        self.matrix = matrix
        self.logic_engine = logic_engine
        self.blocked_nodes = set()

    def register_blocked_node(self, coordinate: Tuple[int, int]) -> None:
        """Register a coordinate as blocked by the safety gate."""
        self.blocked_nodes.add(coordinate)

    def register_inventory_change(self, inventory: List[str]) -> None:
        """Update blocked nodes registry when a key is acquired."""
        has_key = any("key" in item.lower() for item in inventory)
        if has_key:
            # Remove all door nodes from blocked_nodes so they can be path-found
            for node_id, data in self.matrix.graph.nodes(data=True):
                if data.get("node_type") == "SPATIAL" and data.get("tile_type") == "DOOR":
                    pos = data.get("pos")
                    if pos and tuple(pos) in self.blocked_nodes:
                        self.blocked_nodes.remove(tuple(pos))

    def _parse_location(self, loc_str: str) -> Optional[Tuple[int, int]]:
        """Parse a location string like '(r,c)' into a tuple."""
        if not loc_str:
            return None
        match = re.match(r"^\((\d+),(\d+)\)$", loc_str.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def _find_entity_pos(self, target_type: str) -> Optional[Tuple[Tuple[int, int], str]]:
        """Find the first entity of the given type and return its pos and id."""
        for node_id, data in self.matrix.graph.nodes(data=True):
            if data.get("node_type") == "ENTITY" and data.get("entity_type") == target_type:
                loc_str = data.get("location")
                if loc_str:
                    pos = self._parse_location(loc_str)
                    if pos:
                        return pos, node_id
        return None

    def _find_frontier(self, current_pos: Tuple[int, int], inventory: List[str]) -> Optional[Tuple[int, int]]:
        """Find an unexplored frontier spatial node that is reachable."""
        reachable_nodes = []
        for node_id, data in self.matrix.graph.nodes(data=True):
            if data.get("node_type") == "SPATIAL" and data.get("tile_type") == "EMPTY":
                pos = data.get("pos")
                if not pos:
                    continue
                
                # A true frontier node borders at least one coordinate that is entirely
                # missing from the knowledge graph (i.e. currently FOG in the world).
                is_frontier = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor_id = f"Tile_{pos[0]+dr}_{pos[1]+dc}"
                    if not self.matrix.graph.has_node(neighbor_id):
                        is_frontier = True
                        break
                
                if is_frontier and pos != current_pos:
                    reachable_nodes.append(pos)
        
        # Sort by distance (L1 heuristic)
        reachable_nodes.sort(key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
        
        for pos in reachable_nodes:
            path = self.matrix.find_topological_path(current_pos, pos, inventory, exclude_nodes=self.blocked_nodes)
            if path:
                return pos
        return None

    def synthesize_goal_stack(self, current_pos: Tuple[int, int], inventory: List[str]) -> List[SubGoal]:
        """Synthesize a stack of sub-goals to progress in the environment."""
        goal_entity = self._find_entity_pos("GOAL")
        has_key = any("key" in item.lower() for item in inventory)
        
        if goal_entity:
            goal_pos, goal_id = goal_entity
            
            # 1. Try direct path assuming we have NO keys
            direct_path_no_keys = self.matrix.find_topological_path(current_pos, goal_pos, [], exclude_nodes=self.blocked_nodes)
            if direct_path_no_keys:
                return [SubGoal(GoalType.REACH_EXIT, goal_pos, target_entity=goal_id)]
            
            # 2. Try path assuming we have a key
            path_with_key = self.matrix.find_topological_path(current_pos, goal_pos, inventory + ["key_simulated"], exclude_nodes=self.blocked_nodes)
            if path_with_key:
                # Find the door that is blocking us
                door_pos = None
                for pos in path_with_key:
                    node_id = f"Tile_{pos[0]}_{pos[1]}"
                    node_data = self.matrix.graph.nodes.get(node_id, {})
                    if node_data.get("tile_type") == "DOOR":
                        door_pos = pos
                        break
                
                if door_pos:
                    if has_key:
                        return [
                            SubGoal(GoalType.UNLOCK_DOOR, door_pos),
                            SubGoal(GoalType.REACH_EXIT, goal_pos, target_entity=goal_id)
                        ]
                    else:
                        key_entity = self._find_entity_pos("KEY")
                        if key_entity:
                            key_pos, key_id = key_entity
                            return [
                                SubGoal(GoalType.FETCH_KEY, key_pos, target_entity=key_id),
                                SubGoal(GoalType.UNLOCK_DOOR, door_pos),
                                SubGoal(GoalType.REACH_EXIT, goal_pos, target_entity=goal_id)
                            ]
        
        # 3. Fallback: Explore fog frontier
        frontier_pos = self._find_frontier(current_pos, inventory)
        if frontier_pos:
            return [SubGoal(GoalType.EXPLORE_FRONTIER, frontier_pos)]

        # 4. No fog frontier → try unlocking a door to reach new areas
        door_result = self._find_locked_door_frontier(current_pos, inventory, has_key)
        if door_result:
            return door_result

        return []

    def _find_locked_door_frontier(
        self,
        current_pos: Tuple[int, int],
        inventory: List[str],
        has_key: bool,
    ) -> Optional[List[SubGoal]]:
        """Find a locked door and plan to unlock it for further exploration."""
        for node_id, data in self.matrix.graph.nodes(data=True):
            if data.get("node_type") != "SPATIAL" or data.get("tile_type") != "DOOR":
                continue
            pos = data.get("pos")
            if not pos:
                continue

            # Check if we can path to the tile adjacent to the door (using keys)
            # We need to reach a tile next to the door, face it, and toggle.
            path_to_door = self.matrix.find_topological_path(
                current_pos, pos, inventory + ["key_simulated"], exclude_nodes=self.blocked_nodes
            )
            if not path_to_door:
                continue

            if has_key:
                return [
                    SubGoal(GoalType.UNLOCK_DOOR, pos),
                    SubGoal(GoalType.EXPLORE_FRONTIER, pos),
                ]
            else:
                key_entity = self._find_entity_pos("KEY")
                if key_entity:
                    key_pos, key_id = key_entity
                    return [
                        SubGoal(GoalType.FETCH_KEY, key_pos, target_entity=key_id),
                        SubGoal(GoalType.UNLOCK_DOOR, pos),
                        SubGoal(GoalType.EXPLORE_FRONTIER, pos),
                    ]
        return None

    def compile_execution_plan(
        self,
        goal_stack: List[SubGoal],
        current_pos: Tuple[int, int],
        current_dir: Tuple[int, int],
        inventory: List[str]
    ) -> List[str]:
        """Compile a list of primitive actions to execute the goal stack."""
        actions: List[str] = []
        pos = current_pos
        direction = current_dir
        simulated_inv = list(inventory)
        ring = self.matrix._DIR_RING
        
        for goal in goal_stack:
            path = self.matrix.find_topological_path(pos, goal.target_pos, simulated_inv, exclude_nodes=self.blocked_nodes)
            if not path:
                # Cannot reach this sub-goal, abort planning rest
                break
                
            seq: List[str] = []
            try:
                facing_idx = ring.index(tuple(direction))
            except ValueError:
                facing_idx = 0
                
            for i in range(len(path) - 1):
                curr_r, curr_c = path[i]
                next_r, next_c = path[i + 1]
                
                target_dr = next_r - curr_r
                target_dc = next_c - curr_c
                
                try:
                    target_idx = ring.index((target_dr, target_dc))
                except ValueError:
                    continue
                    
                delta = (target_idx - facing_idx) % 4
                if delta == 1:
                    seq.append("TURN_RIGHT")
                elif delta == 3:
                    seq.append("TURN_LEFT")
                elif delta == 2:
                    seq.append("TURN_RIGHT")
                    seq.append("TURN_RIGHT")
                    
                next_node_id = f"Tile_{next_r}_{next_c}"
                next_node_data = self.matrix.graph.nodes.get(next_node_id, {})
                tile_type = next_node_data.get("tile_type", "")
                
                if tile_type in ("DOOR", "INTERACTIVE"):
                    seq.append("TOGGLE_INTERACT")
                    seq.append("MOVE_FORWARD")
                else:
                    seq.append("MOVE_FORWARD")
                    
                facing_idx = target_idx

            actions.extend(seq)
            
            # Update position and direction
            pos = goal.target_pos
            direction = ring[facing_idx]
                
            # Append interaction commands
            if goal.goal_type == GoalType.FETCH_KEY:
                actions.append("PICK_UP")
                simulated_inv.append("key_simulated")
                
        return actions
