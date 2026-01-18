import numpy as np

# ------------------------------------------------------------
# Actions for gridworld
# ------------------------------------------------------------
# 0: Up, 1: Down, 2: Left, 3: Right
ACTIONS = {
    0: (-1, 0),
    1: ( 1, 0),
    2: ( 0,-1),
    3: ( 0, 1),
}
ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}

# ------------------------------------------------------------
# 1) Build an MDP from a 2D mask: S, R, P(a,s,s')
# ------------------------------------------------------------
def build_gridworld_mdp(mask: np.ndarray, terminal_coords=None, slip_prob: float = 0.0):
    H, W = mask.shape
    terminal_coords = set(terminal_coords or [])

    # State space S
    states = [(i, j) for i in range(H) for j in range(W) if not np.isneginf(mask[i, j])]
    state_to_idx = {s: k for k, s in enumerate(states)}
    N = len(states)
    A = len(ACTIONS)

    # Reward per state
    R = np.array([mask[i, j] for (i, j) in states], dtype=float)

    # Transition tensor
    P = np.zeros((A, N, N), dtype=float)

    def next_state_index(i, j, a):
        """Apply action a from (i,j). If hits wall/boundary -> stay."""
        di, dj = ACTIONS[a]
        ni, nj = i + di, j + dj
        if not (0 <= ni < H and 0 <= nj < W) or np.isneginf(mask[ni, nj]):
            return state_to_idx[(i, j)]  # stay
        return state_to_idx[(ni, nj)]

    for (i, j) in states:
        s = state_to_idx[(i, j)]

        if (i, j) in terminal_coords:
            for a in range(A):
                P[a, s, s] = 1.0
            continue

        for a in range(A):
            if slip_prob <= 0.0:
                # Deterministic
                s_next = next_state_index(i, j, a)
                P[a, s, s_next] = 1.0
            else:
                p_intended = 1.0 - slip_prob
                p_other = slip_prob / 3.0

                for a_out in range(A):
                    p = p_intended if a_out == a else p_other
                    s_next = next_state_index(i, j, a_out)
                    P[a, s, s_next] += p

    if not np.allclose(P.sum(axis=2), 1.0):
        raise ValueError("Each (a,s) row of P must sum to 1. Check construction.")

    return states, state_to_idx, R, P


# ------------------------------------------------------------
# 2) Value Iteration for MDP
#    V*(s) = max_a [ R(s) + gamma * sum_{s'} P(s'|s,a) V*(s') ]
# ------------------------------------------------------------
def value_iteration_mdp(P: np.ndarray, R: np.ndarray, gamma: float,
                        tol: float = 1e-8, max_iter: int = 200000):
    A, N, _ = P.shape
    V = np.zeros(N, dtype=float)

    for it in range(max_iter):
        # Q[a, s] = R[s] + gamma * sum_{s'} P[a,s,s'] * V[s']
        Q = np.empty((A, N), dtype=float)
        for a in range(A):
            Q[a] = R + gamma * (P[a] @ V)

        V_new = np.max(Q, axis=0)
        pi = np.argmax(Q, axis=0)

        if np.max(np.abs(V_new - V)) < tol:
            return V_new, pi, it + 1

        V = V_new

    raise RuntimeError("Value iteration did not converge (try smaller gamma or check P).")


# ------------------------------------------------------------
# 3) Helpers: display V and policy on the grid
# ------------------------------------------------------------
def vector_to_grid(mask: np.ndarray, state_to_idx: dict, vec: np.ndarray):
    H, W = mask.shape
    out = np.full((H, W), np.nan, dtype=float)
    for (i, j), idx in state_to_idx.items():
        out[i, j] = vec[idx]
    return out

def policy_to_grid(mask: np.ndarray, state_to_idx: dict, pi: np.ndarray, terminal_coords=None):
    H, W = mask.shape
    terminal_coords = set(terminal_coords or [])
    out = np.full((H, W), "", dtype=object)

    # mark walls
    for i in range(H):
        for j in range(W):
            if np.isneginf(mask[i, j]):
                out[i, j] = "■"  # wall

    for (i, j), idx in state_to_idx.items():
        if np.isneginf(mask[i, j]):
            continue
        if (i, j) in terminal_coords:
            out[i, j] = "T"  # terminal
        else:
            out[i, j] = ARROWS[int(pi[idx])]
    return out


# ------------------------------------------------------------
# 4) Demo / Experiment
# ------------------------------------------------------------
if __name__ == "__main__":

    mask = np.array([
        [ 0, -np.inf, -1, -1, -1],
        [-1, -np.inf, -1, -np.inf, -1],
        [-1, -np.inf, -1, -np.inf, -1],
        [-1, -1,      -1, -np.inf, 101],
        [-3, -3,      -3, -3,      -3],
    ], dtype=float)


    terminals = {(3, 4)}  

    gamma = 0.9


    slip_prob = 0.0

    states, s2i, R, P = build_gridworld_mdp(mask, terminal_coords=terminals, slip_prob=slip_prob)

    V_star, pi_star, iters = value_iteration_mdp(P, R, gamma)

    V_grid = vector_to_grid(mask, s2i, V_star)
    pi_grid = policy_to_grid(mask, s2i, pi_star, terminal_coords=terminals)

    print("\n" + "=" * 70)
    print("MDP Value Iteration (Optimal Control)")
    print("=" * 70)
    print(f"|S| = {len(states)}, gamma = {gamma}, slip_prob = {slip_prob}, iterations = {iters}")

    print("\nOptimal Value Grid V* (walls = nan):")
    print(V_grid)

    print("\nGreedy Policy Grid (■=wall, T=terminal):")
    print(pi_grid)
