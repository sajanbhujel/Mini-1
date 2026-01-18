# Mini-1

This repository contains the code and supporting materials for **Mini-1** of  
**EECS 590: Advanced Topics in Electrical Engineering and Computer Science (Reinforcement Learning)**.

The objective of this mini-project is to model, analyze, and experiment with **finite Markov processes**, including **Markov Reward Processes (MRPs)** and **Markov Decision Processes (MDPs)**, using gridworld environments. The implementation closely follows the theoretical formulation of Markov processes discussed in class.

---

## Repository Contents

This repository includes the following files:

- `thinking_log_1.md`  
  Contains the thinking log used to record ideas, confusions, questions, and plans while working through the assignment (Question 1).

- `Finite_Markov_2.py`  
  Implements abstract base classes and inheritance for finite Markov processes and Markov Reward Processes (Question 2).

- `value_convergence_3.py`  
  Implements iterative value convergence using the Bellman equation, including matrix-based and element-wise updates (Question 3).

- `initialization_2D_mask_4.py`  
  Constructs a gridworld Markov Reward Process from a 2D mask by building the state space, reward vector, and transition matrix (Question 4).

- `policy_6.py`  
  Extends the MRP formulation to a Markov Decision Process by introducing actions and computing optimal value functions and policies using value iteration (Question 6).

- `README.md`  
  This file.

---

## Overview

The project begins by constructing finite Markov processes using clear data structures for states, rewards, and transitions. Value convergence is studied through iterative Bellman updates, and the effect of different discount factors is explored. Gridworld environments are initialized from 2D masks that define rewards and obstacles. Finally, actions are added to transition from MRPs to MDPs, enabling optimal control through value iteration.

---

## How to Run

All scripts are written in Python and require only the NumPy library.

Examples:

```bash
python initialization_2D_mask_4.py
