# Mini-1 Thinking Log

## 1. Ideas
- I think a Markov process is a good way to model grid worlds because the next state depends only on the current state.
- For the grid world, each cell can be treated as a state.
- Obstacles (-∞) probably mean the agent cannot transition into those states.
- Rewards can be taken directly from the 2D mask.

## 2. Confusions
- I am not fully sure how to construct the transition matrix P for edge cells.
- I am confused about whether terminal states should transition to themselves or stop.
- I am not sure when matrix-based value iteration is better than element-wise updates.

## 3. Questions
- Should probabilities in P always sum to 1 even near obstacles?
- How should we treat −∞ rewards numerically in code?
- Is value convergence guaranteed for all γ < 1?

## 4. Plans
- First, implement a simple 3×3 grid world without obstacles.
- Then extend it to the given 2D mask with obstacles.
- Experiment with γ = 0.5, 0.9, and 0.99.
- Compare convergence speed for matrix vs iterative updates.
