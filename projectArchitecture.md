# System Architecture Specification: Neuro-Symbolic Autonomous Mapping Rover

This document serves as the complete technical blueprint for your autonomous room-mapping rover. It bridges physical mechatronic hardware, ROS 2 middleware, and our 4-subsystem **Neuro-Symbolic Cognitive Engine**.

## 1. Complete Hardware & Software Stack Selection

### Physical & Mechatronic Hardware Stack

|**Subsystem**|**Hardware Selection**|**Specification & Rationale**|
|---|---|---|
|**Edge Compute Brain**|Raspberry Pi 5 (8GB RAM)|Runs Ubuntu 24.04 LTS, ROS 2 middleware, and the Python Cognitive Engine. Excellent multi-core CPU performance for graph operations and local vector search.|
|**Low-Level Motor MCU**|ESP32-S3 Development Board|Handles real-time hardware interrupts (encoders), Dual PID motor loops via **FreeRTOS**, and IMU sensor fusion. Communicates with Pi 5 over micro-ROS or high-speed UART (`115200`–`921600 baud`).|
|**Motor Drivers**|BTS7960 High-Power H-Bridge (x2)|Handles up to 43A continuous load, ensuring clean, heat-dissipated PWM power delivery for differential drive motors.|
|**Primary Spatial Sensor**|RPLiDAR A1M8 or LD19 2D LiDAR|$360^\circ$ planar scanning at 10 Hz ($12\text{m}$ range) for 2D Occupancy Grid SLAM and boundary detection.|
|**Semantic & Opening Sensor**|VL53L5CX 8x8 Multi-Zone ToF Array|Measures an $8 \times 8$ grid of distance points ahead. Distinguishes narrow structural gaps (e.g., chair legs) from real doorways ($0.7\text{m}$–$1.2\text{m}$ clear opening).|
|**Odometry & Pose Tracking**|Optical Wheel Encoders + BNO055 9-DOF IMU|Hardware-level quadrature encoding paired with an onboard sensor-fusion IMU to prevent wheel slip drift.|
|**Chassis & Power**|2WD Differential Drive Chassis + 12V LiFePO4 / LiPo Battery|Active front/rear differential drive with caster support. Stepped down via buck converters to 5V/5A for Pi 5/ESP32 and 12V direct to motor drivers.|

### Software Architecture & Framework Stack

```
[ Cognitive Layer ]  ----> Python 3.12 | PyReason | FAISS | CogDB / Kùzu
         │
  (ZeroMQ / gRPC)
         │
[ Middleware Layer]  ----> ROS 2 Humble/Jazzy | SLAM Toolbox | Nav2 Costmaps
         │
   (micro-ROS)
         │
[ Microcontroller ]  ----> FreeRTOS (ESP32-S3) | Quadrature Encoders | PID Motors
```

- **OS / Base System:** Ubuntu 24.04 LTS Server (64-bit).
    
- **Robotics Middleware:** **ROS 2 Humble / Jazzy**. Handles spatial transformations (`tf2`), laser scan aggregation (`sensor_msgs/msg/LaserScan`), and wheel velocity execution (`geometry_msgs/msg/Twist`).
    
- **Low-Level Firmware:** C++ compiled via PlatformIO on VS Code using **FreeRTOS** tasks on the ESP32-S3.
    
- **Cognitive Stack:** Python 3.12 utilizing:
    
    - **PyReason:** For First-Order Open-World Temporal Logic over graph states.
        
    - **FAISS / HNSWLIB:** For in-memory, sub-millisecond vector similarity search in system RAM.
        
    - **CogDB / Kùzu DB:** Disk-backed embedded graph database on the Pi 5's NVMe/MicroSD storage.
        
    - **ZeroMQ / PyZMQ:** Non-blocking asynchronous message bus bridging ROS 2 nodes to the Python Cognitive Loop.
        

## 2. Mapping the Cognitive Subsystems to Physical Robotics

```
                       +---------------------------------------------------+
                       |                 EXECUTIVE ADMIN                   |
                       | - Monitors SLAM Uncertainty (Entropy)             |
                       | - Triggers "Sleep/Consolidation" on Docking/Idle  |
                       +-------------------------+-------------------------+
                                                 |
                   +-----------------------------+-----------------------------+
                   |                                                           |
+------------------v------------------+                     +------------------v------------------+
|      FAST PLASTICITY MEMORY         |                     |       SYMBOLIC LOGIC ENGINE         |
|  (FAISS Vector Index in System RAM) |                     |   (PyReason Rule Verification)      |
| - Caches dynamic 2D grid updates    |                     | - Enforces Doorway & Room Axioms    |
| - Sub-ms spatial lookup             |                     | - Safety Stop: Distance < 0.15m     |
+------------------+------------------+                     +------------------+------------------+
                   |                                                           |
                   +-----------------------------+-----------------------------+
                                                 |
                       +-------------------------v-------------------------+
                       |              CORE KNOWLEDGE MATRIX                |
                       |       (CogDB Graph Database on NVMe Disk)         |
                       | - Nodes: Rooms, Doors, Obstacles                  |
                       | - Edges: CONNECTS_TO, LEADS_TO, BOUNDS            |
                       +---------------------------------------------------+
```

### Subsystem A: Fast Plasticity Memory (In-RAM Buffer)

- **What it stores:** A live, low-latency spatial index (FAISS) holding recent 2D point-cloud feature vectors, active local costmap windows, and recent ToF matrix profiles.
    
- **How it works:** When LiDAR/ToF reads a wall gap, the feature vector is instantly checked against RAM. If it matches a known temporary obstacle (like a moving foot or box), it's handled without querying the disk.
    

### Subsystem B: Symbolic Logic Engine (Domain Knowledge Rules)

- **What it stores:** Hardcoded domain rules written in **PyReason** representing physical space axioms.
    
- **Core Axiom Set:**
    
    - **Doorway Rule:**
        
        $$\text{IF } \text{WallGap}(w) \land (0.7\text{m} \le w \le 1.2\text{m}) \land \text{ToF\_DepthBeyond} > 1.5\text{m} \implies \text{IsDoorway}(x, y) \text{ [Confidence: 0.85, 1.0]}$$
        
    - **Room Completion Axiom:**
        
        $$\text{IF } \text{RaycastCoverage}(Room\_A) \ge 98\% \land \text{UnexploredFrontiers}(Room\_A) = 0 \implies \text{RoomMappingComplete}(Room\_A)$$
        
    - **Hard Safety Override:**
        
        $$\text{IF } \text{MinSensorDistance} < 0.15\text{m} \implies \text{HaltAndRotate}$$
        

### Subsystem C: Core Knowledge Matrix (Persistent Graph DB)

- **What it stores:** A persistent graph (CogDB) written to disk.
    
- **Graph Schema:**
    
    - **Nodes:** `Room(id="Room_1")`, `Doorway(id="Door_1", coords=[2.1, 4.3])`, `Obstacle(type="Unclassified")`.
        
    - **Edges:** `(Room_1) -[CONNECTS_TO]-> (Door_1)`, `(Door_1) -[LEADS_TO]-> (Room_2)`, `(Room_1) -[BOUNDS]-> (Wall_North)`.
        

### Subsystem D: Executive Admin (Metacognitive Loop)

- **What it stores:** System health state, navigation uncertainty, and frontier goal selection.
    
- **Metacognitive Operations:**
    
    1. **Active Mapping Mode:** Reads SLAM map entropy. If entropy is high (robot is lost/confused), it slows linear speed and commands a $360^\circ$ rotation to re-localize.
        
    2. **Sleep & Consolidation Mode:** When the robot reaches a temporary dead-end or completes a room, the Executive Admin pauses exploration for 2–3 seconds. It reviews Fast Memory, runs PyReason consistency checks, merges new rooms/doors into the Core Graph, and clears old RAM buffers.
        

## 3. System Integration & Data Flow Blueprint

```
+---------------------------------------------------------------------------------------+
|                                ROS 2 MIDDLEWARE LAYER                                 |
|                                                                                       |
|  [LiDAR / ToF / Encoders] ──► [SLAM Toolbox] ──► /map & /tf Topics                   |
|                                     │                                                 |
|                                     ▼                                                 |
|                           [ZMQ Bridge ROS Node]                                       |
+-------------------------------------│-------------------------------------------------+
                                      │ (ZeroMQ Socket: Port 5555)
                                      ▼
+---------------------------------------------------------------------------------------+
|                           COGNITIVE ENGINE (Python 3.12)                              |
|                                                                                       |
|  1. Fast Plasticity RAM ──► 2. PyReason Rules ──► 3. Executive Admin Decision         |
|                                                         │                             |
|                                                         ▼                             |
|  [Core Knowledge Graph] ◄── (Sleep Consolidation) ── [Waypoint Selected: (x, y)]      |
+---------------------------------------------------------│-----------------------------+
                                                          │
                                                          ▼ (ZeroMQ Response)
+---------------------------------------------------------------------------------------+
|  [Nav2 Controller Node] ──► /cmd_vel Topic ──► [micro-ROS / ESP32-S3] ──► [Motors]   |
+---------------------------------------------------------------------------------------+
```

1. **Sensors $\rightarrow$ ROS 2:** RPLiDAR and VL53L5CX publish raw point clouds and range arrays to ROS 2 topics (`/scan`, `/tof/grid`).
    
2. **ROS 2 $\rightarrow$ ZeroMQ Bridge:** A lightweight Python ROS node consumes `/map`, `/scan`, and `/tf` (odometry pose), packages them into JSON/MsgPack payloads, and pushes them across ZeroMQ to the Cognitive Engine.
    
3. **Cognitive Reasoning:**
    
    - **Fast Memory** extracts frontier points (unexplored boundaries).
        
    - **PyReason** evaluates if any frontier point matches `IsDoorway` or `HighPriorityExplorationTarget`.
        
    - **Executive Admin** selects the optimal target waypoint `(x, y, theta)`.
        
4. **Execution:** The selected waypoint is sent back over ZeroMQ to the ROS 2 Nav2 stack as a `NavigateToPose` action goal. ROS 2 handles low-level local obstacle avoidance (DWA/TEB local planner) and sends velocity commands (`/cmd_vel`) to the ESP32-S3.
    

## 4. Drawbacks & Technical Mitigations

> **Key Rule for Reliability:** Never mix high-level decision loops with low-level motor safety loops. High-level cognition can pause or lag without causing a collision if low-level hardware safety operates independently.

|**Cons / Drawbacks**|**Impact**|**Technical Solution / Mitigation**|
|---|---|---|
|**Symbol Grounding Noise**|LiDAR/ToF sensor noise makes a wall look like a door or vice-versa.|**Fuzzy Interval Logic in PyReason:** Instead of binary `True/False`, rules evaluate bounds like `[0.75, 0.95]` confidence. Require 3 consecutive matching sensor frames before instantiating a `Doorway` node.|
|**Python Reasoning Latency**|Graph queries and logic checks drop decision rates to 2–5 Hz (Nav2 needs 20+ Hz).|**Dual-Rate Asynchronous Decoupling:** The ESP32 micro-controller runs motor safety loops at **50 Hz** (stops instantly if distance $< 15\text{cm}$). The Python Cognitive Engine runs asynchronously at **2 Hz** generating high-level goal waypoints.|
|**Debugging Complexity**|Hard to trace why the robot chose a specific room or waypoint.|**Structured Explainability Logs:** The Executive Admin outputs a human-readable trace for every goal: `[DECISION] Moving to (x=1.2, y=3.4). Reason: Rule "Doorway_Exploration" satisfied with confidence 0.88. Target Room: Room_2.`|

## 5. Step-by-Step Phased Implementation Roadmap

**1.Phase 1: Mechatronic Hardware & ESP32 Base:**Weeks 1 - 2.

- Assemble chassis, mount motors, BTS7960 drivers, wheel encoders, battery, and ESP32-S3.
    
- Flash ESP32 firmware with **FreeRTOS** tasks: Task 1 handles quadrature encoder decoding; Task 2 runs Dual PID motor control at 50 Hz.
    
- Verify precise dead-reckoning movement by commanding forward $1.0\text{m}$ and $90^\circ$ turns via serial terminal.
    

**2.Phase 2: ROS 2 Infrastructure & SLAM Setup:**Weeks 3 - 4.

- Mount Pi 5, RPLiDAR A1, IMU, and VL53L5CX ToF sensor to chassis. Install Ubuntu 24.04 & ROS 2.
    
- Implement `micro-ROS` or serial bridge node between ESP32 and Pi 5 to publish `/odom` and consume `/cmd_vel`.
    
- Configure `SLAM Toolbox` to build a clean 2D occupancy grid map driven manually via teleop joystick.
    

**3.Phase 3: Cognitive Engine Core Development:**Weeks 5 - 6.

- Build standalone Python modules for:
    
    1. **Fast Plasticity:** FAISS vector index caching frontier points in RAM.
        
    2. **Core Graph:** CogDB schema instantiating `Room`, `Door`, and `Obstacle` nodes.
        
    3. **Symbolic Engine:** PyReason rule script containing doorway detection and room-enclosure logic.
        
- Test the engine in pure software using simulated occupancy grid maps.
    

**4.Phase 4: Integration & ZeroMQ Bridge:**Weeks 7 - 8.

- Build the ZeroMQ bridge node connecting ROS 2 topic streams to the Python Cognitive Loop.
    
- Implement the **Executive Admin** loop: pipeline sensory inputs through FAISS and PyReason, generate goal waypoints, and dispatch them to ROS 2 Nav2.
    
- Test real-world autonomous exploration in a multi-room environment.
    

**5.Phase 5: Benchmarking & Comparative Analysis:**Weeks 9 - 10.

- Run controlled mapping experiments across 3 system variants:
    
    1. **Baseline 1:** Default ROS 2 Nav2 Frontier Exploration.
        
    2. **Baseline 2:** LLM/VLM Agent Controller querying web API.
        
    3. **Our System:** Neuro-Symbolic Cognitive Engine.
        
- Measure and record: Total time to complete mapping ($s$), total distance traveled ($m$), decision latency ($ms$), and path efficiency.