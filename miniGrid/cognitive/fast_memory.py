"""
fast_memory.py — Fast Plasticity Memory Layer (Subsystem A)
============================================================
Implements a fast episodic memory layer using FAISS for
vectorized state similarity and novelty detection. This layer
acts as a short-term RAM, allowing the agent to recall recent
experiences and measure the novelty of current states.

Target: Python 3.10
"""

from __future__ import annotations

import collections
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class EpisodicExperience:
    """Represents a single step of episodic memory."""
    experience_id: str
    vector: np.ndarray
    state_dict: Dict[str, Any]
    action: str
    reward: float
    timestamp: float
    novelty_score: float


# ──────────────────────────────────────────────
# State Vectorization
# ──────────────────────────────────────────────

class StateVectorizer:
    """Converts observations and logic state into a normalized vector.
    
    Produces a fixed 64-dimensional L2-normalized float32 vector
    representing the current agent state for similarity matching.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        
        # Maps integer direction enum to (dx, dy)
        # 0: NORTH, 1: EAST, 2: SOUTH, 3: WEST
        self._dir_map = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }

    def vectorize(self, obs: Dict[str, Any], state_context: Dict[str, Any]) -> np.ndarray:
        """Constructs an L2-normalized 64-dim embedding of the state."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        
        ps = obs.get("player_state", {})
        px, py = ps.get("position", (0, 0))
        direction_idx = ps.get("direction", 0)
        dx, dy = self._dir_map.get(direction_idx, (0, 0))
        health = ps.get("health", 100) / 100.0  # Normalize to [0, 1]
        
        inventory = ps.get("inventory", [])
        inv_count = len(inventory)
        
        # Populate early dimensions
        vec[0] = float(px)
        vec[1] = float(py)
        vec[2] = float(dx)
        vec[3] = float(dy)
        vec[4] = float(health)
        vec[5] = float(inv_count)
        
        # Contextual boolean flags
        vec[6] = 1.0 if state_context.get("is_wall", False) else 0.0
        vec[7] = 1.0 if state_context.get("is_hazard", False) else 0.0
        vec[8] = 1.0 if state_context.get("is_door", False) else 0.0
        vec[9] = 1.0 if state_context.get("is_locked", False) else 0.0
        vec[10] = 1.0 if state_context.get("has_key", False) else 0.0
        
        # Target tile encoding (hash-based to spread activation slightly)
        target_tile = state_context.get("target_tile", "UNKNOWN")
        tile_hash = hash(target_tile) % 10
        vec[11 + tile_hash] = 1.0
        
        # Ensure it is at least 1-dimensional row vector (1, 64) for FAISS
        vec = vec.reshape(1, -1)
        
        # L2 Normalize
        faiss.normalize_L2(vec)
        
        return vec


# ──────────────────────────────────────────────
# Memory Layer
# ──────────────────────────────────────────────

class FastPlasticityMemory:
    """Episodic memory store and novelty detector using FAISS."""

    def __init__(self, dimension: int = 64, capacity: int = 1000) -> None:
        """Initialise FAISS index and ring buffer for episodic memory.
        
        Args:
            dimension: Dimensionality of the state vector.
            capacity: Maximum number of experiences to retain.
        """
        self.dimension = dimension
        self.capacity = capacity
        
        # FAISS index (using IDMap to associate vectors with our IDs)
        # Note: IndexIDMap2 supports removing vectors, which is useful for ring buffers.
        self._base_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index = faiss.IndexIDMap2(self._base_index)
        
        self.experience_buffer: collections.deque[EpisodicExperience] = collections.deque(maxlen=self.capacity)
        self.vectorizer = StateVectorizer(dimension=self.dimension)
        
        self._counter = 0

    def calculate_novelty(self, vector: np.ndarray, k: int = 3) -> float:
        """Calculate novelty score ΔE based on local vector neighborhood.
        
        Args:
            vector: (1, D) numpy array L2-normalized state vector.
            k: Number of nearest neighbors to average over.
            
        Returns:
            Novelty score bounded [0.0, 1.0+]. Higher means more novel.
        """
        if self.faiss_index.ntotal < k:
            return 1.0
            
        distances, _ = self.faiss_index.search(vector, k)
        # Distances are squared L2 since we use IndexFlatL2
        # Average distance to k-nearest neighbors represents novelty
        mean_l2 = float(np.mean(distances[0]))
        
        return mean_l2

    def store_experience(
        self,
        obs: Dict[str, Any],
        state_context: Dict[str, Any],
        action: str,
        reward: float,
        k: int = 3
    ) -> EpisodicExperience:
        """Vectorize state, compute novelty, and store in memory.
        
        Args:
            obs: Raw environment observation.
            state_context: Forward tile logic context.
            action: Action executed.
            reward: Reward received.
            k: Neighborhood size for novelty calculation.
            
        Returns:
            The created EpisodicExperience object.
        """
        vec = self.vectorizer.vectorize(obs, state_context)
        
        novelty = self.calculate_novelty(vec, k=k)
        
        exp_id_str = f"exp_{self._counter}_{uuid.uuid4().hex[:8]}"
        self._counter += 1
        
        # State snapshot for persistence
        ps = obs.get("player_state", {})
        state_dict = {
            "position": tuple(ps.get("position", (0, 0))),
            "direction": ps.get("direction", 0),
            "inventory": ps.get("inventory", []),
            "target_tile": state_context.get("target_tile", "UNKNOWN")
        }
        
        exp = EpisodicExperience(
            experience_id=exp_id_str,
            vector=vec.copy(),
            state_dict=state_dict,
            action=action,
            reward=reward,
            timestamp=time.time(),
            novelty_score=novelty
        )
        
        # Enforce ring buffer behavior in FAISS if at capacity
        if len(self.experience_buffer) == self.capacity:
            oldest_exp = self.experience_buffer[0]
            # ID mappings are integers in FAISS, we use the buffer index as ID
            pass # Handling exact FAISS removal is slow; for this implementation we use ID lists
            
        # Add to FAISS. We map FAISS integer IDs to the deque index / object
        faiss_id = np.array([self._counter], dtype=np.int64)
        self.faiss_index.add_with_ids(vec, faiss_id)
        
        # We also store the FAISS ID on the object so we can retrieve it
        exp._faiss_id = self._counter
        
        # Maintain sliding window in FAISS manually if needed, or let FAISS grow slightly 
        # For a true strict capacity, we remove the oldest ID
        if len(self.experience_buffer) == self.capacity:
            oldest_exp = self.experience_buffer[0]
            old_faiss_id = np.array([oldest_exp._faiss_id], dtype=np.int64)
            self.faiss_index.remove_ids(old_faiss_id)
            
        self.experience_buffer.append(exp)
        return exp

    def query_similar(
        self,
        vector: np.ndarray,
        k: int = 3
    ) -> List[Tuple[EpisodicExperience, float]]:
        """Retrieve the k most similar past experiences to a state vector.
        
        Args:
            vector: (1, D) numpy array L2-normalized state vector.
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of (EpisodicExperience, L2_distance) tuples.
        """
        if self.faiss_index.ntotal == 0:
            return []
            
        k_search = min(k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(vector, k_search)
        
        results = []
        for dist, f_id in zip(distances[0], indices[0]):
            if f_id == -1:
                continue
            
            # Find the experience in the deque matching this FAISS ID
            # In a production system, a dict mapping ID -> Exp is faster (O(1))
            # but for N=1000, linear scan is fine, or we can use a dict.
            matched_exp = None
            # Fast reverse scan since recent memories are more likely matched
            for exp in reversed(self.experience_buffer):
                if getattr(exp, "_faiss_id", -1) == f_id:
                    matched_exp = exp
                    break
                    
            if matched_exp:
                results.append((matched_exp, float(dist)))
                
        return results

    def get_recent_experiences(self, n: int = 10) -> List[EpisodicExperience]:
        """Return the most recently stored experiences.
        
        Args:
            n: Number of experiences to retrieve.
            
        Returns:
            List of up to n EpisodicExperience objects in chronological order
            (oldest first within the returned window).
        """
        if len(self.experience_buffer) == 0:
            return []
            
        n = min(n, len(self.experience_buffer))
        # Convert the tail of the deque to a list
        return list(collections.deque(self.experience_buffer, maxlen=n))
