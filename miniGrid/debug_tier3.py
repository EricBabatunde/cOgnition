import main
from environment.custom_rpg_env import CustomRPGEnv
from cognitive.executive_admin import ExecutiveGoalEngine
from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.fast_memory import FastPlasticityMemory
from cognitive.symbolic_engine import SymbolicLogicEngine
import sys

env = CustomRPGEnv(tier=3)
matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
matrix.load_graph("config/graph_memory.json")
symbolic_engine = SymbolicLogicEngine()
symbolic_engine.load_rules_from_config("config/innate_instincts.json")
fast_memory = FastPlasticityMemory(dimension=64, capacity=1000)
goal_engine = ExecutiveGoalEngine(matrix, symbolic_engine)

obs, info = env.reset(seed=42)
main.update_knowledge_from_obs(matrix, obs, prev_pos=None, env=env)

action_queue = []
last_action = ""
for step in range(500):
    ps = obs["player_state"]
    pos = tuple(ps["position"])
    state_context = main.get_forward_tile_context(env, obs)
    vec = fast_memory.vectorizer.vectorize(obs, state_context)
    current_novelty = fast_memory.calculate_novelty(vec)
    
    if current_novelty >= 0.50 and last_action != "TOGGLE_INTERACT":
        action_queue.clear()
        
    if not action_queue:
        goal_stack = goal_engine.synthesize_goal_stack(pos, ps.get("inventory", []))
        if goal_stack:
            action_queue = goal_engine.compile_execution_plan(
                goal_stack, pos, main._DIR_TO_VEC[main.Direction(ps["direction"])], ps.get("inventory", [])
            )
            print(f"Step {step} @ {pos} Planned: {', '.join(g.goal_type.name for g in goal_stack)} ({len(action_queue)} actions)")
        if not action_queue:
            action_queue.append("TURN_RIGHT")
            
    act_name = action_queue.pop(0)
    print(f"Step {step} @ {pos} (Inv: {ps.get('inventory', [])}) Action: {act_name}")
    action = main._ACTION_NAME_MAP.get(act_name)
    last_action = act_name
    
    # Z3 block
    if action == main.Action.MOVE_FORWARD:
        is_safe, _, _, _ = symbolic_engine.verify_action_dynamic("MOVE_FORWARD", state_context)
        if not is_safe:
            print(f"Z3 BLOCKED MOVE_FORWARD at {pos}")
            target_pos = state_context.get("target_pos")
            if target_pos:
                if state_context.get("is_door"):
                    node_id = f"Tile_{target_pos[0]}_{target_pos[1]}"
                    if node_id in matrix.graph:
                        matrix.graph.nodes[node_id]["is_locked"] = True
                else:
                    goal_engine.register_blocked_node(target_pos)
            action_queue.clear()
            continue
            
    obs, r, term, trunc, info = env.step(action)
    goal_engine.register_inventory_change(obs["player_state"].get("inventory", []))
    main.update_knowledge_from_obs(matrix, obs, prev_pos=pos, env=env)
    
    if term or trunc: break
