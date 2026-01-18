import numpy as np

# Matrix-based value iteration

def value_iteration_matrix(P, R, gamma, tol=1e-8, max_iter=100000):
    V = np.zeros(len(R))
    for _ in range(max_iter):
        V_new = R + gamma * (P @ V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    raise RuntimeError("Value iteration did not converge.")

# Element-wise value iteration

def value_iteration_elementwise(P, R, gamma, tol=1e-8, max_iter=100000):
    N = len(R)
    V = np.zeros(N)
    for _ in range(max_iter):
        V_new = np.zeros(N)
        for s in range(N):
            V_new[s] = R[s] + gamma * np.sum(P[s] * V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    raise RuntimeError("Value iteration did not converge.")

