miniGrid/
├── main.py
├── requirements.txt
└── cognitive/
    ├── __init__.py
    ├── fast_memory.py      # Subsystem A: RAM Layer (FAISS/Hebbian cache)
    ├── symbolic_engine.py  # Subsystem B: Rule & Logic Layer (Z3/First-Order Logic)
    ├── core_graph.py       # Subsystem C: Persistent Graph Matrix
    └── executive_admin.py  # Subsystem D: Metacognitive Supervisor & Event Loop