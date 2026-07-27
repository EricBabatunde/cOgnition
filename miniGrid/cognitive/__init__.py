"""
cognitive — Neuro-Symbolic Cognitive Engine Package
====================================================
Exposes the core subsystems used by the Executive Admin to
perceive, reason, and act within the RPG environment.
"""

from .core_graph import CoreKnowledgeMatrix
from .symbolic_engine import SymbolicLogicEngine

__all__ = [
    "CoreKnowledgeMatrix",
    "SymbolicLogicEngine",
]
