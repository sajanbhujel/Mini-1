import numpy as np

# ------------------------------------------------------------
# 1) Build S, R, P from a 2D mask
# ------------------------------------------------------------
def build_gridworld_mrp(mask: np.ndarray, terminal_coords=None):
    H, W = mask.shape
    terminal_coords = set(terminal_coords or [])

    # S: valid states = all non-wall cells
    states = [(i, j) for i in range(H) for j in range(W) if not np.isneginf(mask[i, j])]
    state_to_idx = {s: k for k, s in enumerate(states)}
    N = len(states)

    # R: reward per state
    R = np.array([mask[i, j] for (i, j) in states], dtype=float)

    # P: transitions (MRP, no actions): choose a direction uniformly among 4 moves
    P = np.zeros((N, N), dtype=float)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    for (i, j) in states:
        s = state_to_idx[(i, j)]

        # Optional: absorbing terminal states
        if (i, j) in terminal_coords:
            P[s, s] = 1.0
            continue

        # Otherwise: random walk among 4 directions
        for di, dj in moves:
            ni, nj = i + di, j + dj

            # If out of bounds or wall -> stay put
            if not (0 <= ni < H and 0 <= nj < W) or np.isneginf(mask[ni, nj]):
                P[s, s] += 1.0 / 4.0
            else:
                P[s, state_to_idx[(ni, nj)]] += 1.0 / 4.0

    # Sanity check: each row sums to 1
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Each row of transition matrix P must sum to 1.")

    return states, state_to_idx, R, P


# ------------------------------------------------------------
# 2) Value convergence for MRP: V = R + gamma P V
# ------------------------------------------------------------
def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float, tol: float = 1e-8, max_iter: int = 100000):
    """
    Iteratively solve V = R + gamma * P @ V
    """
    V = np.zeros_like(R, dtype=float)

    for it in range(max_iter):
        V_new = R + gamma * (P @ V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, it + 1
        V = V_new

    raise RuntimeError("Value iteration did not converge (try smaller gamma or check P).")



def values_to_grid(mask: np.ndarray, state_to_idx: dict, V: np.ndarray):
    H, W = mask.shape
    V_grid = np.full((H, W), np.nan, dtype=float)
    for (i, j), idx in state_to_idx.items():
        V_grid[i, j] = V[idx]
    return V_grid


# ------------------------------------------------------------
# 4) Example usage + experiments ()
# ------------------------------------------------------------
if __name__ == "__main__":
    mask1 = np.array([
        [ 0, -np.inf, -1, -1, -1],
        [-1, -np.inf, -1, -np.inf, -1],
        [-1, -np.inf, -1, -np.inf, -1],
        [-1, -1,      -1, -np.inf, -1],
        [-3, -3,      -3, -3,      -3],
    ], dtype=float)


    mask2 = mask1.copy()
    mask2[3, 4] = 101  

    masks = [("PromptMask", mask1), ("GoalMask", mask2)]


    terminals = None

    gammas = [0.1, 0.3, 0.6, 0.9, 0.99]

    for name, mask in masks:
        print("\n" + "=" * 70)
        print(f"Mask: {name}")
        print("=" * 70)

        states, state_to_idx, R, P = build_gridworld_mrp(mask, terminal_coords=terminals)

        print(f"Number of states |S| = {len(states)}")
        print(f"R shape = {R.shape}, P shape = {P.shape}")

        for gamma in gammas:
            V, iters = value_iteration(P, R, gamma)
            V_grid = values_to_grid(mask, state_to_idx, V)

            print(f"\nGamma={gamma}   iterations={iters}")
            print("Value Grid (walls = nan):")
            print(V_grid)
