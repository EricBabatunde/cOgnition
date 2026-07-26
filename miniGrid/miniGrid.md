# Architecture Specification: Neuro-Symbolic Mini RPG Sandbox

## 1. High-Level Architecture Overview

```
                                  +-----------------------------------------------+
                                  |            EXECUTIVE ADMIN                    |
                                  |  - Tracks Prediction Error (ΔE)               |
                                  |  - Manages Goal State & Exploration Strategy  |
                                  |  - Triggers Reflection / Graph Consolidation  |
                                  +-----------------------+-----------------------+
                                                          |
                      +-----------------------------------+-----------------------------------+
                      |                                                                       |
+---------------------v---------------------+                           +---------------------v---------------------+
|      FAST PLASTICITY MEMORY               |                           |       SYMBOLIC LOGIC ENGINE               |
|      (RAM Buffer / Vector Index)          |                           |       (PyReason Logic Engine)             |
|  - Caches last N local FOV matrices       |                           |  - Evaluates First-Order / Temporal Rules |
|  - Sub-ms state/pattern similarity        |                           |  - Safety overrides & Action candidate    |
+---------------------+---------------------+                           +---------------------+---------------------+
                      |                                                                       |
                      +-----------------------------------+-----------------------------------+
                                                          |
                                  +-----------------------v-----------------------+
                                  |        CORE KNOWLEDGE MATRIX                  |
                                  |     (NetworkX / CogDB Graph Model)            |
                                  |  - Persistent Spatial Graph (Map Memory)      |
                                  |  - Causal Edges (Item -> Effect)              |
                                  |  - Pre-seeded via innate_instincts.json       |
                                  +-----------------------+-----------------------+
                                                          |
                                                          v
+-------------------------------------------------------------------------------------------------------------------+
|                                                 VISUAL UI LAYER                                                   |
|                                                                                                                   |
|  +-------------------------------------+-----------------------------------------------------------------------+  |
|  |     TERMINAL DASHBOARD (Rich)       |                       WEB MIND INSPECTOR (PyVis)                      |  |
|  |  - Panel 1: World View (Fog of War) |  - Renders interactive node-edge graph on "Sleep/Reflection" cycles.  |  |
|  |  - Panel 2: Fast Memory & Logic     |  - Shows real-time evolution of spatial maps & causal rules.          |  |
|  |  - Panel 3: Thought Stream Log      |                                                                       |  |
|  +-------------------------------------+-----------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------------------------------------------+
```

## 2. Project File & Module Structure (Antigravity IDE Layout)

Plaintext

```
rpg_cognitive_engine/
├── config/
│   ├── innate_instincts.json      # Pre-fed core axioms (e.g., doors, health, keys)
│   └── game_config.json           # Grid size, hazard damage, vision range
├── environment/
│   ├── __init__.py
│   ├── custom_rpg_env.py          # Gymnasium-compatible Mini RPG Grid engine
│   └── entities.py                # Agent, Key, Door, Hazard, Lever, Chest definitions
├── cognitive/
│   ├── __init__.py
│   ├── fast_memory.py             # RAM buffer for short-term FOV history
│   ├── symbolic_engine.py         # PyReason logic solver & rule evaluator
│   ├── core_graph.py              # Persistent Knowledge Matrix (Graph storage)
│   └── executive_admin.py         # Metacognitive supervisor & prediction error loop
├── ui/
│   ├── __init__.py
│   ├── terminal_dashboard.py      # Split-panel UI using Python Rich
│   └── web_inspector.py           # PyVis graph visualizer (exports to graph_mind.html)
├── main.py                        # Execution entry point & turn-based loop
└── requirements.txt               # Dependencies (rich, pyvis, pyreason, networkx, gymnasium)
```

## 3. Subsystem Breakdown & Operational Specifications

### A. Environment Engine (`custom_rpg_env.py`)

- **State Mechanics:**
    
    - Grid Size: $10 \times 10$ tile arena.
        
    - Partial Observability ("Fog of War"): Agent receives a $5 \times 5$ forward-facing egocentric vision cone. Unseen tiles remain masked in fog (`░░`).
        
    - Player Attributes: `Position (x,y)`, `Orientation (N/S/E/W)`, `Health (100)`, `Inventory (List)`.
        
- **Action Space (Realistic Grid Controls):**
    
    0. `TURN_LEFT` ($90^\circ$ counter-clockwise)
    
    1. `TURN_RIGHT` ($90^\circ$ clockwise)
        
    2. `MOVE_FORWARD` (1 tile in facing direction)
        
    3. `PICK_UP` (Item on current tile)
        
    4. `TOGGLE_INTERACT` (Door/Lever on adjacent forward tile)
        

### B. Cognitive Engine Subsystems

#### 1. Fast Plasticity Memory (`fast_memory.py`)

- **Role:** Acts as the agent's short-term working buffer (RAM).
    
- **Storage:** Stores the last 20 steps of egocentric FOV snapshots, raw observations, and recent action choices.
    
- **Function:** Evaluates spatial similarity—if the agent enters a room configuration similar to one visited 5 steps ago, it alerts the Executive Admin of potential looping/backtracking.
    

#### 2. Symbolic Logic Engine (`symbolic_engine.py`)

- **Role:** Evaluates First-Order Logic rules over the immediate FOV and active inventory.
    
- **PyReason Rule Examples:**
    
    $$\text{IF } \text{InFront}(\text{Door\_Red}) \land \text{HasItem}(\text{Key\_Red}) \implies \text{CanPerform}(\text{TOGGLE\_INTERACT}) \text{ [Confidence: 1.0]}$$
    
    $$\text{IF } \text{InFront}(\text{Hazard\_Lava}) \implies \text{ForbidAction}(\text{MOVE\_FORWARD}) \text{ [Confidence: 1.0]}$$
    

#### 3. Core Knowledge Matrix (`core_graph.py`)

- **Role:** Persistent global memory storing spatial topography and causal mechanics.
    
- **Graph Structure:**
    
    - **Spatial Nodes:** `Tile(2,3)`, `Room_1`, `Door_Main`.
        
    - **Entity/Item Nodes:** `Key_Red`, `Chest_Gold`, `Lever_A`.
        
    - **Edges:** `CONNECTS_TO`, `CONTAINS`, `OPENS`, `CAUSES_DAMAGE`.
        
- **Initialization:** Bootstrapped on startup by parsing `innate_instincts.json`.
    

#### 4. Executive Admin (`executive_admin.py`)

- **Role:** The metacognitive supervisor driving the decision loop.
    
- **Prediction Error Loop ($\Delta E$):**
    
    1. Formulates expectation before action: $E_{\text{expected}} = (\text{New\_Pose}, \text{Expected\_HP}, \text{Expected\_State})$.
        
    2. Executes action and observes environment step: $E_{\text{actual}}$.
        
    3. Calculates Prediction Error: $\Delta E = \vert{}E_{\text{actual}} - E_{\text{expected}}\vert{}$.
        
    4. **Reflection Trigger:** If $\Delta E > 0$, execution pauses briefly for a **Sleep/Reflection Phase**:
        
        - The Executive Admin queries the Symbolic Engine to isolate the root cause.
            
        - New causal edges (e.g., `Lever_A` $\rightarrow$ `DISARMS` $\rightarrow$ `Trap_1`) are permanently written to the Core Knowledge Matrix.
            

### C. Interface Layer Specifications

#### 1. Terminal Dashboard (`terminal_dashboard.py`)

Uses `Rich.Live` and `Rich.Layout` to render a 3-panel display in the terminal:

Plaintext

```
+------------------------------------+------------------------------------+
|                                    | FAST MEMORY & LOGIC                |
|           WORLD VIEW               | - Active Goal: Retrieve Key_Red    |
|        (Fog of War 10x10)          | - FAISS Match: Room_1 (0.94)       |
|                                    | - PyReason Rule: SafeToStep = True |
|  ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░     +------------------------------------+
|  ░░  .  .  .  .  . ░░ ░░ ░░ ░░     | THOUGHT STREAM LOG                 |
|  ░░  .  ▲  .  🚪  . ░░ ░░ ░░ ░░     | [01] Obs: Door_Red in FOV          |
|  ░░  .  .  .  .  . ░░ ░░ ░░ ░░     | [02] Querying Graph: Key_Red location|
|  ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░     | [03] Plan: Move South to Tile(2,1) |
|                                    | [04] Predict Error ΔE: 0.0 (OK)    |
+------------------------------------+------------------------------------+
| Stats: HP: 100 | Step: 14 | Inv: [Key_Red] | State: EXPLORING           |
+-------------------------------------------------------------------------+
```

#### 2. Web Mind Inspector (`web_inspector.py`)

- Uses `PyVis` to generate an interactive HTML file (`graph_mind.html`).
    
- Whenever the Executive Admin triggers a Reflection Phase, `web_inspector.py` writes out the updated graph structure.
    
- Nodes are color-coded: **Blue** = Explored Rooms/Tiles, **Gold** = Items/Keys, **Red** = Hazards, **Green** = Learned Causal Relations.
    

## 4. Innate Instincts Config File (`innate_instincts.json`)

This JSON file pre-feeds the Core Knowledge Matrix with fundamental world instincts:

JSON

```
{
  "innate_rules": [
    {
      "rule_id": "wall_blocking",
      "premise": "InFront(Wall)",
      "conclusion": "Forbid(MOVE_FORWARD)",
      "confidence": 1.0
    },
    {
      "rule_id": "goal_priority",
      "premise": "SeeInFOV(GoalChest)",
      "conclusion": "SetPriorityTarget(GoalChest)",
      "confidence": 0.95
    },
    {
      "rule_id": "locked_door_requires_key",
      "premise": "InFront(Door) AND IsLocked(Door)",
      "conclusion": "RequiresItem(MatchingKey)",
      "confidence": 0.90
    },
    {
      "rule_id": "hazard_avoidance",
      "premise": "InFront(Hazard)",
      "conclusion": "Forbid(MOVE_FORWARD)",
      "confidence": 1.0
    }
  ]
}
```

## 5. Turn-Based Step Cycle Workflow

```
[1. Env Step] ──► [2. Parse FOV & Update Fast RAM] ──► [3. Graph Query & Symbolic Verification]
                                                                        │
[6. Render Rich UI & Update Web Graph] ◄── [5. Exec Admin Check ΔE] ◄── [4. Execute Chosen Action]
```

1. **Observe:** Environment outputs egocentric $5 \times 5$ vision matrix and player status.
    
2. **Cache:** Fast Memory receives raw FOV snapshot and updates short-term spatial trail.
    
3. **Reason:** Core Graph maps visible tiles into spatial nodes. Symbolic Logic Engine evaluates candidate actions against innate instincts and graph rules, selecting the safest, highest-priority move.
    
4. **Act:** Action is dispatched to `custom_rpg_env.py`.
    
5. **Reflect:** Executive Admin compares new state against expectations. If prediction error $\Delta E > 0$, it pauses for graph consolidation and writes new edges.
    
6. **Display:** `terminal_dashboard.py` updates the split-screen view, and `web_inspector.py` refreshes the graph file.


Here is a 6-phase development and testing plan designed to build, stress-test, and integrate our custom RPG sandbox and Cognitive Engine step by step.

**1.Phase 1: Environment Engine & Rich Terminal UI:**Phase 1.

**Goal:** Build a working 2D grid environment and visual terminal dashboard with zero AI logic involved.

- **Build:**
    
    - `environment/custom_rpg_env.py`: $10 \times 10$ tile map, player state (pose, inventory, health), fog-of-war vision mask ($5 \times 5$ forward cone), and basic tile types (wall, door, key, hazard, goal).
        
    - `ui/terminal_dashboard.py`: Split-panel layout using `Rich.Live` to display the game board, player telemetry, and dummy log output.
        
- **Validation Test:**
    
    - **Manual Teleop Run:** Drive the agent using keyboard arrows (`W/A/S/D`). Verify that moving into walls blocks movement, stepping on hazards reduces health, picking up keys updates inventory, fog of war unmasks correctly as the agent rotates, and the `Rich` UI renders cleanly at 20+ FPS without flickering.
        

**2.Phase 2: Core Knowledge Matrix & Web Mind Inspector:**Phase 2.

**Goal:** Establish persistent memory storage and interactive graph visualization.

- **Build:**
    
    - `config/innate_instincts.json`: Pre-seed basic world rules (e.g., walls block movement, keys open matching doors).
        
    - `cognitive/core_graph.py`: NetworkX/CogDB graph wrapper to parse the JSON config, create spatial tile nodes, and append typed edges (`CONNECTS_TO`, `CONTAINS`, `REQUIRES`).
        
    - `ui/web_inspector.py`: Expose a function `export_graph_to_html()` using `PyVis`.
        
- **Validation Test:**
    
    - **Synthetic Graph Test:** Write a unit test script that feeds fake step coordinates (`Tile_1_1` $\rightarrow$ `Tile_1_2`) into `core_graph.py`. Run `web_inspector.py` and open `graph_mind.html` in a browser. Verify that spatial nodes link correctly, innate rules populate on startup, and node colors match entity types.
        

**3.Phase 3: Fast Plasticity Memory Buffer:**Phase 3.

**Goal:** Construct the short-term working RAM memory for FOV snapshots and loop detection.

- **Build:**
    
    - `cognitive/fast_memory.py`: In-RAM rolling buffer storing the last 20 $5 \times 5$ vision grids and player actions. Include a matrix similarity lookup to detect when the agent is repeating states or running in circles.
        
- **Validation Test:**
    
    - **Loop Detection Test:** Script a repeating 4-step motion pattern (`Move North` $\rightarrow$ `Turn Right` $\rightarrow$ `Move East` $\rightarrow$ `Turn Right` $\dots$). Verify that `fast_memory.py` correctly calculates high spatial similarity ($> 90\%$) and raises a `LoopDetected` flag to signal the engine to change tactics.
        

**4.Phase 4: Symbolic Logic Engine Integration:**Phase 4.

**Goal:** Implement deterministic rule evaluation and action candidate filtering.

- **Build:**
    
    - `cognitive/symbolic_engine.py`: PyReason wrapper that evaluates incoming local FOV observations against innate graph rules and returns allowed/forbidden action masks.
        
- **Validation Test:**
    
    - **Rule Filtering Test:** Place the agent directly in front of a locked red door without a key. Query `symbolic_engine.py` and assert that `TOGGLE_INTERACT` is marked as `FORBIDDEN`. Place `Key_Red` in the agent's inventory, re-query, and assert that `TOGGLE_INTERACT` transitions to `ALLOWED` with confidence 1.0. Repeat for hazard tiles to verify collision/damage avoidance rules.
        

**5.Phase 5: Executive Admin, Prediction Error, & Reflection Loop:**Phase 5.

**Goal:** Connect all subsystems into a unified turn cycle managed by the Executive Admin.

- **Build:**
    
    - `cognitive/executive_admin.py`: The metacognitive loop that formulates state expectations, measures Prediction Error ($\Delta E = \vert{}E_{\text{actual}} - E_{\text{expected}}\vert{}$), and triggers the **Sleep / Reflection Phase** when unexpected events occur.
        
    - `main.py`: The main loop orchestrating Environment $\rightarrow$ Fast Memory $\rightarrow$ Logic Engine $\rightarrow$ Executive Admin $\rightarrow$ UI Refresh.
        
- **Validation Test:**
    
    - **Unmapped Hazard Surprise Test:** Place an unmapped hidden trap on a tile (not listed in `innate_instincts.json`). Let the agent step on it.
        
    - **Expected Result:** The agent takes unexpected damage ($HP$ drops from 100 to 80). The Executive Admin measures $\Delta E > 0$, pauses the game cycle, prints `[REFLECTION PHASE]` in the Rich log, queries the Symbolic Engine to isolate `Action(Step) + Tile(HiddenTrap)`, and writes a permanent new rule edge to the Core Graph (`Tile_X_Y` $\rightarrow$ `CAUSES_DAMAGE`). The web inspector updates immediately with the new hazard node.
        

**6.Phase 6: Full Autonomous Play & Strategy Benchmarking:**Phase 6.

**Goal:** Let the agent navigate and solve complete, multi-room mini RPG levels completely unassisted.

- **Execution:**
    
    - Run full episodes across 3 difficulty tiers:
        
        1. _Tier 1:_ Single room with key, locked door, and goal chest.
            
        2. _Tier 2:_ Multi-room dungeon with fog of war, multiple keys, and static lava hazards.
            
        3. _Tier 3:_ Dynamic dungeon with hidden pressure plates, dynamic levers, and decoy chests.
            
- **Validation Test:**
    
    - Track metrics: Steps to completion, total prediction errors ($\Delta E$), graph growth rate, and zero-shot adaptation (verifying it never makes the same mistake twice).