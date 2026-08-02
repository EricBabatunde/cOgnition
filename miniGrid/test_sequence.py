import main
from environment.custom_rpg_env import CustomRPGEnv
from cognitive.executive_admin import ExecutiveGoalEngine
from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine
import networkx as nx

env = CustomRPGEnv(tier=3)
matrix = CoreKnowledgeMatrix("config/innate_instincts.json")

# Simulating Run 1 end state
matrix.add_spatial_node(4, 9, "EMPTY", explored=True)
matrix.graph.nodes["Tile_4_9"]["is_locked"] = False
matrix.add_entity_node("Key_red_15_3", "KEY", {"location": "(15,3)", "color": "red"})

# Room C to Room D corridor is unlocked (blue door)
matrix.add_spatial_node(14, 9, "EMPTY", explored=True)
matrix.graph.nodes["Tile_14_9"]["is_locked"] = False

# Room D to Room B corridor barrier removed by lever
matrix.add_spatial_node(9, 14, "EMPTY", explored=True)
matrix.add_spatial_node(10, 14, "EMPTY", explored=True)

# Path around: (4, 8) -> (4, 7) -> ... -> (9, 5) [Room A to Room C] -> (14, 9) [Room C to Room D] -> (9, 14) [Room D to Room B] -> (4, 10)
# Add all nodes in the path so they exist in the graph!
for r, c in [(4, 8), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (9, 6), (9, 5), (10, 5), (11, 5), (12, 5), (13, 5), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (13, 14), (12, 14), (11, 14), (10, 14), (9, 14), (8, 14), (7, 14), (6, 14), (5, 14), (4, 14), (4, 13), (4, 12), (4, 11), (4, 10)]:
    matrix.add_spatial_node(r, c, "EMPTY", explored=True)
    matrix.add_typed_edge(f"Tile_{r}_{c}", f"Tile_{r}_{c}", "ADJACENT") # Just to make it simple

# Start Run 2
obs, info = env.reset(seed=42)

env.unwrapped._player.position = [4, 8]
env.unwrapped._player.direction = 1 # EAST
env.unwrapped._update_explored()
obs = env.unwrapped._get_obs()

main.update_knowledge_from_obs(matrix, obs, prev_pos=(4, 7), env=env)

# Now check path
path = matrix.find_topological_path((4, 8), (4, 10), [])
print("Path to (4, 10):", path)
if path:
    print("Path length:", len(path))
