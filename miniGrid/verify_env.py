#!/usr/bin/env python3
"""
verify_env.py — Phase 1 Environment Verification Script
========================================================
Validates that all core subsystem primitives required by the
Neuro-Symbolic RPG Cognitive Engine are installed and functional.

Targets: Python 3.10
Subsystems Under Test:
    A. Gymnasium / MiniGrid  — Environment engine
    B. Z3-Solver             — Symbolic logic / SMT solver
    C. FAISS (faiss-cpu)     — Fast vector similarity index
    D. NetworkX              — Persistent knowledge graph
"""

from __future__ import annotations

import sys
import time
import traceback

# ──────────────────────────────────────────────
# ANSI helpers for structured terminal output
# ──────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

PASS = f"{_GREEN}✓ PASS{_RESET}"
FAIL = f"{_RED}✗ FAIL{_RESET}"

def _header(title: str) -> None:
    width = 62
    print(f"\n{_CYAN}{'━' * width}{_RESET}")
    print(f"{_CYAN}┃{_RESET} {_BOLD}{title}{_RESET}")
    print(f"{_CYAN}{'━' * width}{_RESET}")

def _log(msg: str) -> None:
    print(f"  {_DIM}│{_RESET}  {msg}")

def _result(name: str, passed: bool, detail: str = "") -> bool:
    tag = PASS if passed else FAIL
    suffix = f"  {_DIM}({detail}){_RESET}" if detail else ""
    print(f"  {_DIM}└─{_RESET} [{tag}] {name}{suffix}")
    return passed


# ──────────────────────────────────────────────
# 0. Version Manifest
# ──────────────────────────────────────────────
def print_version_manifest() -> bool:
    _header("DEPENDENCY VERSION MANIFEST")
    all_ok = True

    packages = {
        "gymnasium":  "gymnasium",
        "minigrid":   "minigrid",
        "faiss-cpu":  "faiss",
        "z3-solver":  "z3",
        "networkx":   "networkx",
    }

    for label, module_name in packages.items():
        try:
            mod = __import__(module_name)
            # z3 exposes version via z3.get_version_string()
            if module_name == "z3":
                ver = mod.get_version_string()
            elif module_name == "faiss":
                # faiss-cpu doesn't have a simple __version__; fall back
                ver = getattr(mod, "__version__", "build-ok (no __version__)")
            else:
                ver = getattr(mod, "__version__", "unknown")
            _log(f"{label:16s}  →  {ver}")
        except ImportError as exc:
            _log(f"{label:16s}  →  {_RED}IMPORT FAILED: {exc}{_RESET}")
            all_ok = False

    _result("All imports resolved", all_ok)
    return all_ok


# ──────────────────────────────────────────────
# 1. Gymnasium / MiniGrid Environment Init
# ──────────────────────────────────────────────
def verify_gymnasium() -> bool:
    _header("SUBSYSTEM A — Gymnasium / MiniGrid Environment")

    try:
        import gymnasium as gym
        import minigrid  # noqa: F401  — registers envs on import

        env_id = "MiniGrid-Empty-8x8-v0"
        _log(f"Creating environment: {env_id}")
        env = gym.make(env_id)

        _log("Calling env.reset() ...")
        obs, info = env.reset(seed=42)
        _log(f"Observation space : {env.observation_space}")
        _log(f"Action space      : {env.action_space}")
        _log(f"Obs 'image' shape : {obs['image'].shape}")
        _log(f"Reset info keys   : {list(info.keys())}")

        # Sanity: take one no-op step
        obs2, reward, terminated, truncated, info2 = env.step(env.action_space.sample())
        _log(f"Post-step obs ok  : shape={obs2['image'].shape}, reward={reward}")

        env.close()
        return _result("MiniGrid env init + reset + step", True, env_id)

    except Exception as exc:
        traceback.print_exc()
        return _result("MiniGrid env init + reset + step", False, str(exc))


# ──────────────────────────────────────────────
# 2. Z3 SMT Solver — Trivial Safety Rule
# ──────────────────────────────────────────────
def verify_z3() -> bool:
    _header("SUBSYSTEM B — Z3 SMT Solver (Symbolic Logic)")

    try:
        from z3 import Bool, Solver, And, Implies, sat

        _log("Defining boolean safety rule ...")
        _log("  Rule: InFront(Hazard) ∧ ¬HasShield → Forbid(MOVE_FORWARD)")

        hazard_ahead  = Bool("InFront_Hazard")
        has_shield    = Bool("HasShield")
        forbid_move   = Bool("Forbid_MOVE_FORWARD")

        # Encode: (hazard_ahead AND NOT has_shield) => forbid_move
        safety_rule = Implies(And(hazard_ahead, has_shield == False), forbid_move)

        solver = Solver()
        solver.add(safety_rule)

        # Assert scenario: hazard IS ahead, shield IS absent
        solver.add(hazard_ahead == True)
        solver.add(has_shield  == False)

        _log("Checking satisfiability ...")
        result = solver.check()
        _log(f"Solver result     : {result}")

        if result == sat:
            model = solver.model()
            forbid_val = model.evaluate(forbid_move)
            _log(f"Forbid(MOVE_FWD)  : {forbid_val}")
            ok = bool(forbid_val)  # Should be True
            return _result("Z3 safety rule evaluation", ok,
                           f"Forbid={forbid_val}")
        else:
            return _result("Z3 safety rule evaluation", False, "unsat")

    except Exception as exc:
        traceback.print_exc()
        return _result("Z3 safety rule evaluation", False, str(exc))


# ──────────────────────────────────────────────
# 3. FAISS — IndexFlatL2 Instantiation
# ──────────────────────────────────────────────
def verify_faiss() -> bool:
    _header("SUBSYSTEM C — FAISS Vector Index (Fast Memory)")

    try:
        import numpy as np
        import faiss

        dim = 128  # Embedding dimension for FOV snapshots
        _log(f"Creating IndexFlatL2 (d={dim}) ...")
        index = faiss.IndexFlatL2(dim)

        # Insert 10 synthetic FOV embeddings
        n_vectors = 10
        rng = np.random.default_rng(seed=42)
        data = rng.random((n_vectors, dim)).astype(np.float32)
        index.add(data)
        _log(f"Vectors indexed   : {index.ntotal}")

        # Query with the first vector — should return itself as nearest
        query = data[0:1]
        distances, indices = index.search(query, k=3)
        _log(f"Top-3 neighbours  : indices={indices[0].tolist()}, "
             f"distances={[f'{d:.4f}' for d in distances[0].tolist()]}")

        # Nearest neighbour should be the query itself (distance ≈ 0)
        ok = (indices[0][0] == 0) and (distances[0][0] < 1e-5)
        return _result("FAISS index add + search", ok,
                       f"ntotal={index.ntotal}, nn_dist={distances[0][0]:.6f}")

    except Exception as exc:
        traceback.print_exc()
        return _result("FAISS index add + search", False, str(exc))


# ──────────────────────────────────────────────
# 4. NetworkX — Knowledge Graph Instantiation
# ──────────────────────────────────────────────
def verify_networkx() -> bool:
    _header("SUBSYSTEM D — NetworkX Knowledge Graph")

    try:
        import networkx as nx

        _log("Constructing synthetic spatial knowledge graph ...")
        G = nx.DiGraph()

        # Spatial topology
        G.add_node("Tile_1_1", kind="tile",   explored=True)
        G.add_node("Tile_1_2", kind="tile",   explored=True)
        G.add_node("Tile_2_2", kind="tile",   explored=False)
        G.add_node("Door_Red", kind="door",   locked=True)
        G.add_node("Key_Red",  kind="item",   subtype="key")
        G.add_node("Hazard_Lava", kind="hazard", damage=20)

        G.add_edge("Tile_1_1", "Tile_1_2", relation="CONNECTS_TO")
        G.add_edge("Tile_1_2", "Door_Red", relation="CONNECTS_TO")
        G.add_edge("Tile_1_1", "Key_Red",  relation="CONTAINS")
        G.add_edge("Key_Red",  "Door_Red", relation="OPENS")
        G.add_edge("Tile_2_2", "Hazard_Lava", relation="CONTAINS")
        G.add_edge("Hazard_Lava", "Tile_2_2", relation="CAUSES_DAMAGE")

        _log(f"Nodes             : {G.number_of_nodes()}")
        _log(f"Edges             : {G.number_of_edges()}")

        # Query: what does Key_Red open?
        key_targets = [v for u, v, d in G.out_edges("Key_Red", data=True)
                       if d.get("relation") == "OPENS"]
        _log(f"Key_Red opens     : {key_targets}")

        # Query: which tiles have hazards?
        hazard_tiles = [u for u, v, d in G.edges(data=True)
                        if d.get("relation") == "CONTAINS"
                        and G.nodes[v].get("kind") == "hazard"]
        _log(f"Hazard tiles      : {hazard_tiles}")

        ok = (G.number_of_nodes() == 6
              and G.number_of_edges() == 6
              and key_targets == ["Door_Red"]
              and hazard_tiles == ["Tile_2_2"])
        return _result("NetworkX graph build + query", ok,
                       f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    except Exception as exc:
        traceback.print_exc()
        return _result("NetworkX graph build + query", False, str(exc))


# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────
def main() -> int:
    print(f"\n{_BOLD}{'═' * 62}{_RESET}")
    print(f"{_BOLD}  NEURO-SYMBOLIC RPG ENGINE — ENVIRONMENT VERIFICATION{_RESET}")
    print(f"{_BOLD}  Python {sys.version.split()[0]}  •  {time.strftime('%Y-%m-%d %H:%M:%S')}{_RESET}")
    print(f"{_BOLD}{'═' * 62}{_RESET}")

    results = []
    results.append(print_version_manifest())
    results.append(verify_gymnasium())
    results.append(verify_z3())
    results.append(verify_faiss())
    results.append(verify_networkx())

    # ── Summary ──
    total   = len(results)
    passed  = sum(results)
    failed  = total - passed

    print(f"\n{_BOLD}{'═' * 62}{_RESET}")
    if failed == 0:
        print(f"  {_GREEN}{_BOLD}ALL {total} SUBSYSTEMS VERIFIED SUCCESSFULLY{_RESET}")
        print(f"  {_DIM}Environment is ready for Phase 1 development.{_RESET}")
    else:
        print(f"  {_RED}{_BOLD}{failed}/{total} SUBSYSTEM(S) FAILED{_RESET}")
        print(f"  {_DIM}Review errors above before proceeding.{_RESET}")
    print(f"{_BOLD}{'═' * 62}{_RESET}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
