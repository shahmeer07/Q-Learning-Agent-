# 🤖 Q-Learning Agent Solving a Custom GridWorld  
### A Reinforcement Learning Experiment with Jump Tiles, Obstacles & Dynamic Reward Shaping  
**© 2025 — Shahmeer Khan**

---

![Banner](https://dummyimage.com/1200x260/0b3d91/ffffff&text=Q-Learning+GridWorld+Reinforcement+Learning)

<div align="center">

🧭 **Custom GridWorld Environment** • 🤖 **Q-Learning Agent** • 📊 **Value Function Visualization**  
**Dynamic Rewards, Jump Tiles, Obstacles & Multiple Learning Rates**

</div>

---

# 📘 Overview

This repository contains a complete reinforcement learning experiment where a **Q-Learning agent** learns to navigate a **5×5 GridWorld** with:

✔ Obstacles  
✔ A start & goal state  
✔ Penalty for invalid moves  
✔ A bonus jump tile (`J → JT`)  
✔ Positive terminal reward  
✔ Negative step cost  
✔ Multiple learning rates (α) comparison  
✔ Policy visualization with `matplotlib`  

The goal is to study how learning rate impacts convergence, optimal paths, Q-values, and training stability.

---

# 🗺 GridWorld Layout

### Environment Features

| Element | Description |
|--------|-------------|
| **S** | Start at `(1, 0)` |
| **G** | Goal at `(4, 4)` — reward `+10` |
| **J → JT** | Jump tile: stepping on `(1,3)` teleports to `(3,3)` with reward `+5` |
| **X** | Obstacles at `(2,1)` and `(3,1)` |
| **Actions** | Up, Down, Left, Right (4 discrete actions) |
| **Step Reward** | `-1` |
| **Invalid Move** | No movement + penalty `-1` |

---

# 🚀 Q-Learning Algorithm

The agent uses the standard Q-Learning update rule:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma \max_a Q(s',a) - Q(s,a)\big]
\]

Hyperparameters:

```python
alpha   # learning rate
gamma   = 0.7   # discount factor
epsilon = 0.1   # exploration rate (decays over time)
```

### 🧠 Features Implemented
## ✔ 1. Custom GridWorld Class

Validity checking

Reward shaping

Jump tile

Goal detection

Obstacles

## ✔ 2. Q-Learning Trainer

ε-greedy policy

Early stopping

Reward tracking

Q-table learning for multiple α values

## ✔ 3. Policy Simulation

Once trained, the agent runs deterministically using:

action = np.argmax(Q[state])


Produces:

Optimal path

Total reward

## ✔ 4. Visualizations

Using matplotlib, the project generates:

📈 Rewards vs. Episodes
📊 Heatmap of state-values
🧭 Arrows showing optimal policy
🟩 Highlighting start, goal, obstacles, and jump tiles

### 📂 Repository Files
# File	Purpose
gridworld_q_learning.py	Full environment + Q-Learning + visualization script
# 🔍 Code Summary
GridWorld definition
self.rows = 5
self.cols = 5
self.start = (1, 0)
self.goal = (4, 4)
self.jump_from = (1, 3)
self.jump_to = (3, 3)
self.obstacles = [(2, 1), (3, 1)]

Q-Learning training loop
Q[state][action] += alpha * (
    reward + gamma * max(Q[next_state]) - Q[state][action]
)

Visualizing value & policy
values = np.max(Q, axis=2)
policy  = np.argmax(Q, axis=2)
ax.imshow(values, cmap='viridis')
ax.quiver(...)

📊 Example Outputs
📈 Reward curves for α ∈ {1.0, 0.5, 0.1}

Observe:

α = 1.0 → unstable, high variance

α = 0.5 → fast learning, moderate stability

α = 0.1 → slow but smooth convergence

## 🧭 Optimal policy arrows

# Each grid cell shows:

Value estimate

Arrow for best action

Special markings: S, G, J, JT, X

🚀 Running the Experiment
python gridworld_q_learning.py


Dependencies:

numpy
matplotlib


Install with:

pip install numpy matplotlib

## 🧩 Future Enhancements

Deep Q-Learning (DQN) version

Stochastic wind / cliff variants

Multi-agent comparison

SARSA implementation

OpenAI Gym wrapper

## 📜 License

MIT License
© 2025 — Shahmeer Khan
