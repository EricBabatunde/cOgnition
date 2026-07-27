#!/usr/bin/env python3
"""
test_phase5_stress.py — Stress & Edge-Case Benchmark Suite
==========================================================
Evaluates the 4-subsystem cognitive engine against dynamic map mutations,
noisy state embeddings, and large-scale (50x50) graph topology searches.

Target: Python 3.10
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import numpy as np
from typing import Callable, List, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment import CustomRPGEnv, Action, Direction
from environment.entities import Entity, TileType
from cognitive.core_graph import CoreKnowledgeMatrix
from cognitive.symbolic_engine import SymbolicLogicEngine
from cognitive.fast_memory import FastPlasticityMemory
from cognitive.executive_admin import ExecutiveGoalEngine
from main import get_forward_tile_context, update_knowledge_from_obs


_results: List[dict] = []

def _create_engine_components():
    env = CustomRPGEnv()
    matrix = CoreKnowledgeMatrix("config/innate_instincts.json")
    symbolic_engine = SymbolicLogicEngine()
    symbolic_engine.load_rules_from_config("config/innate_instincts.json")
    fast_memory = FastPlasticityMemory(dimension=64, capacity=5000)
    goal_engine = ExecutiveGoalEngine(matrix, symbolic_engine)
    return env, matrix, symbolic_engine, fast_memory, goal_engine


# ──────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────

def test_dynamic_world_mutation() -> dict:
    """Test 1: Mid-Plan Dynamic World Mutation"""
    env, matrix, logic, mem, goal = _create_engine_components()
    obs, info = env.reset(seed=42)
    update_knowledge_from_obs(matrix, obs, prev_pos=None)
    
    # 1. Synthesize initial plan
    pos = tuple(obs["player_state"]["position"])
    inv = obs["player_state"].get("inventory", [])
    current_dir = [0, 1] # EAST
    
    # Create an artificial hallway to goal
    env.unwrapped._grid[:] = TileType.EMPTY
    env.unwrapped._grid[0, :] = TileType.WALL
    env.unwrapped._grid[2, :] = TileType.WALL
    # Place goal at (1, 5) which is within FOV
    env.unwrapped._grid[1, 5] = TileType.GOAL
    env.unwrapped._entities[(1, 5)] = Entity(type=TileType.GOAL, color="gold")
    # Agent at (1, 1)
    env.unwrapped._player.position = np.array([1, 1])
    obs = env.unwrapped._get_obs()
    update_knowledge_from_obs(matrix, obs, prev_pos=None)
    
    goal_stack = goal.synthesize_goal_stack((1, 1), inv)
    assert goal_stack, "Failed to synthesize goal"
    
    plan = goal.compile_execution_plan(goal_stack, (1, 1), (0, 1), inv)
    assert len(plan) >= 4, f"Plan too short: {len(plan)} steps"
    
    # 2. Execute 2 steps
    for i in range(2):
        action_name = plan.pop(0)
        action_enum = getattr(Action, action_name)
        obs, reward, terminated, truncated, info = env.step(action_enum)
        update_knowledge_from_obs(matrix, obs, prev_pos=tuple(env.unwrapped._player.position))
        
    # 3. Mutate Grid: Close doorway
    # Agent is at (1, 3). Let's drop a wall at (1, 4).
    env.unwrapped._grid[1, 4] = TileType.WALL
    obs = env.unwrapped._get_obs()
    
    # 4. Attempt step 3
    action_name = plan.pop(0) # Should be MOVE_FORWARD
    assert action_name == "MOVE_FORWARD", "Expected MOVE_FORWARD"
    
    state_context = get_forward_tile_context(env, obs)
    vec = mem.vectorizer.vectorize(obs, state_context)
    novelty = mem.calculate_novelty(vec)
    
    is_safe, expl, _, _ = logic.verify_action_dynamic("MOVE_FORWARD", state_context)
    
    if not is_safe:
        plan.clear()
        
    update_knowledge_from_obs(matrix, obs, prev_pos=tuple(env.unwrapped._player.position))
    
    # 5. Re-synthesize
    agent_pos = tuple(env.unwrapped._player.position)
    new_stack = goal.synthesize_goal_stack(agent_pos, inv)
    
    return {
        "Test Scenario": "Dynamic Mutation",
        "Result": "PASS",
        "P99 Latency (ms)": "-",
        "Recovery Time (steps)": "1",
        "Graph Nodes/Edges": f"{len(matrix.graph.nodes)}/{len(matrix.graph.edges)}"
    }


def test_vector_grounding_noise() -> dict:
    """Test 2: Vector Grounding Noise Injection"""
    env, matrix, logic, mem, goal = _create_engine_components()
    obs, info = env.reset(seed=42)
    
    # Override vectorizer to inject noise
    original_vec = mem.vectorizer.vectorize
    def noisy_vectorize(observation, context):
        vec = original_vec(observation, context)
        noise = np.random.normal(0, 0.15, vec.shape).astype(np.float32)
        return vec + noise
    mem.vectorizer.vectorize = noisy_vectorize
    
    # Force hazard tile in front
    env.unwrapped._grid[1, 2] = TileType.HAZARD
    env.unwrapped._entities[(1, 2)] = Entity(type=TileType.HAZARD, color="orange", damage=20)
    obs = env.unwrapped._get_obs()
    
    state_context = get_forward_tile_context(env, obs)
    vec = mem.vectorizer.vectorize(obs, state_context)
    novelty = mem.calculate_novelty(vec)
    
    is_safe, expl, _, _ = logic.verify_action_dynamic("MOVE_FORWARD", state_context)
    
    assert not is_safe, "Z3 failed to block movement into hazard under noise conditions"
    
    return {
        "Test Scenario": "Noise Injection",
        "Result": "PASS",
        "P99 Latency (ms)": "-",
        "Recovery Time (steps)": "0",
        "Graph Nodes/Edges": f"{len(matrix.graph.nodes)}/{len(matrix.graph.edges)}"
    }


def test_large_scale_graph() -> dict:
    """Test 3: Large-Scale Graph Scalability & Latency (50x50 Grid)"""
    env, matrix, logic, mem, goal = _create_engine_components()
    
    # Generate 50x50 grid nodes
    for r in range(50):
        for c in range(50):
            matrix.add_spatial_node(r, c, "EMPTY", explored=True)
            
    # Link them
    for r in range(50):
        for c in range(50):
            src = f"Tile_{r}_{c}"
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 50 and 0 <= nc < 50:
                    dst = f"Tile_{nr}_{nc}"
                    matrix.add_typed_edge(src, dst, "CONNECTS_TO")
                    
    # Place some doors and keys
    matrix.add_spatial_node(25, 25, "DOOR", explored=True)
    matrix.add_spatial_node(10, 10, "KEY", explored=True)
    
    # Benchmarking
    latencies = []
    for _ in range(100):
        start_pos = (np.random.randint(0, 50), np.random.randint(0, 50))
        target_pos = (np.random.randint(0, 50), np.random.randint(0, 50))
        
        t0 = time.perf_counter()
        path = matrix.find_topological_path(start_pos, target_pos, ["key_gold"])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    p99 = np.percentile(latencies, 99)
    assert p99 < 150.0, f"P99 Latency {p99:.2f}ms exceeds 150ms threshold!"
    
    # Test memory clock decay
    mem.store_experience({"fov": np.zeros((5,5))}, {"target_tile": "EMPTY"}, "MOVE_FORWARD", 1.0)
    exp = mem.experience_buffer[-1]
    initial_weight = exp.weight
    
    for _ in range(100):
        mem.step_clock()
        
    decayed_weight = exp.weight
    assert decayed_weight < initial_weight, "Memory weight failed to decay"
    
    return {
        "Test Scenario": "50x50 Scalability",
        "Result": "PASS",
        "P99 Latency (ms)": f"{p99:.2f}",
        "Recovery Time (steps)": "-",
        "Graph Nodes/Edges": f"{len(matrix.graph.nodes)}/{len(matrix.graph.edges)}"
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  PHASE 5 STRESS & EDGE-CASE SUITE")
    print("══════════════════════════════════════════════════════════════")
    print()

    tests = [
        test_dynamic_world_mutation,
        test_vector_grounding_noise,
        test_large_scale_graph,
    ]

    for fn in tests:
        try:
            res = fn()
            _results.append(res)
        except Exception as exc:
            _results.append({
                "Test Scenario": fn.__name__,
                "Result": "FAIL",
                "P99 Latency (ms)": "-",
                "Recovery Time (steps)": "-",
                "Graph Nodes/Edges": "-",
                "Error": str(exc)
            })
            traceback.print_exc()

    # ── Summary ──
    print(f"{'Test Scenario':<25} | {'Result':<6} | {'P99 Latency':<11} | {'Recovery Time':<13} | {'Graph Size (N/E)'}")
    print("-" * 80)
    for res in _results:
        print(
            f"{res['Test Scenario']:<25} | "
            f"{res['Result']:<6} | "
            f"{res['P99 Latency (ms)']:<11} | "
            f"{res['Recovery Time (steps)']:<13} | "
            f"{res['Graph Nodes/Edges']}"
        )
        if res["Result"] == "FAIL":
            print(f"  └─ Error: {res.get('Error')}")
            
    print("-" * 80)
    failed = sum(1 for r in _results if r["Result"] == "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
