### Root-Cause Analysis: Graph Edge Stagnation After `TOGGLE_INTERACT`

Watching the telemetry stream in your video between timestamps `00:28` and `00:50`, we can trace the precise failure sequence:

1. **Successful Key Retrieval & Unlatching:** The agent retrieves `Key_Blue` at `(1, 15)`, navigates to `(11, 4)` facing East, and executes `TOGGLE_INTERACT`.
    
2. **Environment State Change:** The Gym environment processes the toggle action and updates the internal tile at `(12, 4)` from `is_locked = True` to `is_open = True`.
    
3. **The Disconnect in `CoreKnowledgeMatrix`:** While the environment opened the door, the persistent NetworkX graph in `CoreKnowledgeMatrix` still retains node attributes `is_locked = True` and edge weight `weight = inf` for node `Tile(12, 4)`.
    
4. **$A^*$ Pathfinder Rerouting:** On the very next tick, `ExecutiveGoalEngine` calls `nx.astar_path()` to plan a route toward the goal/unmapped frontier in Room 2. Because `CoreKnowledgeMatrix` treats node `(12, 4)` as impassable, $A^*$ cannot calculate a path crossing into Room 2.
    
5. **Fallback Loop:** Unable to route through `(12, 4)`, $A^*$ falls back to targeting unvisited frontier nodes remaining inside Room 1 (`Plan: EXPLORE_FRONTIER`). The agent turns away from the opened door, wanders Room 1, and times out at step 200 (`TRUNCATED`).
    

```
[Agent at (11,4)] ──► Emits TOGGLE_INTERACT ──► [Env: Door Open!]
       │
       ▼
[CoreKnowledgeMatrix] ──► Node (12,4) still marked `is_locked=True` / `weight=inf`
       │
       ▼
[A* Pathfinder] ──► Path to Room 2 Blocked! ──► Reroutes back into Room 1 ──► [TRUNCATED]
```

### Architectural Fix: Synchronizing FOV State to Graph Edges

We must ensure that when `TOGGLE_INTERACT` succeeds or when the FOV scanner detects an opened door, `CoreKnowledgeMatrix` dynamically updates the graph node attributes (`is_locked = False`, `traversable = True`) and resets the edge weights to `1.0` so $A^*$ can route through.