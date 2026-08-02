import main
from environment.custom_rpg_env import CustomRPGEnv
from cognitive.executive_admin import ExecutiveGoalEngine, SubGoal, GoalType
from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine

env = CustomRPGEnv(tier=3)
matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
symbolic_engine = SymbolicLogicEngine()
goal_engine = ExecutiveGoalEngine(matrix, symbolic_engine)

obs, info = env.reset(seed=42)
for r in range(9):
    for c in range(12):
        env.unwrapped._explored[r, c] = True

obs = env.unwrapped._get_obs()
main.update_knowledge_from_obs(matrix, obs, prev_pos=None, env=env)

# Lock (4, 9)
matrix.graph.nodes["Tile_4_9"]["is_locked"] = True

frontier_pos = goal_engine._find_frontier((4, 8), [])
print("Frontier pos:", frontier_pos)
if frontier_pos:
    path = matrix.find_topological_path((4, 8), frontier_pos, [])
    print("Path:", path)

