import main
from environment.custom_rpg_env import CustomRPGEnv
from cognitive.executive_admin import ExecutiveGoalEngine, SubGoal, GoalType
from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine
import networkx as nx

matrix = CoreKnowledgeMatrix("config/innate_instincts.json")

# Manually build just enough graph
matrix.add_spatial_node(4, 8, "EMPTY")
matrix.add_spatial_node(4, 9, "DOOR")
matrix.graph.nodes["Tile_4_9"]["is_locked"] = True
matrix.add_typed_edge("Tile_4_8", "Tile_4_9", "ADJACENT")

# Simulate having the key in graph
matrix.add_entity_node("Key_red_15_3", "KEY", {"location": "(15,3)", "color": "red"})
matrix.add_typed_edge("Tile_4_9", "Door_red_4_9", "CONTAINS")

symbolic_engine = SymbolicLogicEngine()
goal_engine = ExecutiveGoalEngine(matrix, symbolic_engine)

goals = [
    SubGoal(GoalType.UNLOCK_DOOR, (4, 9)),
    SubGoal(GoalType.EXPLORE_FRONTIER, (4, 9))
]

actions = goal_engine.compile_execution_plan(goals, (4, 8), (0, 1), ["key_red"])
print("Actions:", actions)

