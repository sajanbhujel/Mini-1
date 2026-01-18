from abc import ABC, abstractmethod
import numpy as np

# --------------------------------------------------
# Abstract Base Class: Finite Markov Process
# --------------------------------------------------
class FiniteMarkovProcess(ABC):
    def __init__(self, states, P):
        self.states = states
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.P = np.array(P, dtype=float)

        self._validate_transition_matrix()

    def _validate_transition_matrix(self):
        # Each row of P should sum to 1
        row_sums = self.P.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of transition matrix P must sum to 1.")

    @abstractmethod
    def evaluate(self):
        """Evaluate the process (implemented in subclasses)."""
        pass


# --------------------------------------------------
# Derived Class: Markov Reward Process (MRP)
# --------------------------------------------------
class MarkovRewardProcess(FiniteMarkovProcess):
    def __init__(self, states, P, R, gamma):
        super().__init__(states, P)
        self.R = np.array(R, dtype=float)
        self.gamma = gamma

    def evaluate(self, tol=1e-8, max_iter=100000):
        V = np.zeros(len(self.states))

        for _ in range(max_iter):
            V_new = self.R + self.gamma * (self.P @ V)
            if np.max(np.abs(V_new - V)) < tol:
                return V_new
            V = V_new

        raise RuntimeError("Value iteration did not converge.")
