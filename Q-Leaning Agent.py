import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.start = (1, 0)
        self.goal = (4, 4)
        self.jump_from = (1, 3)
        self.jump_to = (3, 3)
        self.obstacles = [(2, 1), (3, 1)]
        self.actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W
        self.action_symbols = ['↑', '↓', '→', '←']

    def is_valid(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols and state not in self.obstacles

    def step(self, state, action_idx):
        dr, dc = self.actions[action_idx]
        next_state = (state[0] + dr, state[1] + dc)
        if not self.is_valid(next_state):
            return state, -1, False
        if next_state == self.jump_from:
            return self.jump_to, 5, False
        if next_state == self.goal:
            return next_state, 10, True
        return next_state, -1, False

    def reset(self):
        return self.start

def train_q_learning(env, episodes=100, alpha=0.1, gamma=0.7, epsilon=0.1):
    Q = np.zeros((env.rows, env.cols, len(env.actions)))
    recent_rewards = []
    episode_rewards = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < 1000:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(Q[state[0], state[1]])
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            best_next = np.max(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += alpha * (reward + gamma * best_next - Q[state[0], state[1], action])
            state = next_state
            steps += 1
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 30:
            recent_rewards = recent_rewards[-30:]
        if len(recent_rewards) == 30 and np.mean(recent_rewards) > 10:
            print(f"Early stop at episode {ep+1}, avg reward: {np.mean(recent_rewards):.2f}")
            break
        epsilon = max(0.01, epsilon * 0.995)
    return Q, episode_rewards

def simulate_policy(env, Q):
    state = env.reset()
    path = [state]
    total_r = 0
    done = False
    for _ in range(100):
        action = np.argmax(Q[state[0], state[1]])
        next_state, r, done = env.step(state, action)
        total_r += r
        path.append(next_state)
        state = next_state
        if done:
            break
    print("Optimal Path:", path)
    print("Total Reward:", total_r)

def visualize_state_values(env, Q, alpha):
    values = np.max(Q, axis=2)
    policy = np.argmax(Q, axis=2)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(values, cmap='viridis')
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    # Arrows for policy
    for i in range(5):
        for j in range(5):
            if (i,j) in env.obstacles or (i,j) == env.goal:
                continue
            action = policy[i,j]
            dx, dy = env.actions[action][1], -env.actions[action][0]  # Quiver: x,y dir (invert y for plot)
            ax.quiver(j, i, dx*0.3, dy*0.3, angles='xy', scale_units='xy', scale=1, color='black')
    # Labels
    for i in range(5):
        for j in range(5):
            if (i,j) == env.start:
                label = 'S'
            elif (i,j) == env.goal:
                label = 'G'
            elif (i,j) == env.jump_from:
                label = 'J'
            elif (i,j) == env.jump_to:
                label = 'JT'
            elif (i,j) in env.obstacles:
                label = 'X'
            else:
                label = f'{values[i,j]:.1f}'
            ax.text(j, i, label, ha='center', va='center', color='white' if values[i,j] < values.mean() else 'black')
    plt.colorbar(im)
    plt.title(f'State Values (Max Q) and Policy (alpha={alpha})')
    plt.show()

# Run for multiple alphas
env = GridWorld()
alphas = [1.0, 0.5, 0.1]
for alpha in alphas:
    print(f"\nRunning with alpha={alpha}")
    Q, rewards = train_q_learning(env, alpha=alpha)
    plt.plot(rewards)
    plt.title(f'Rewards Over Episodes (alpha={alpha})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    simulate_policy(env, Q)
    visualize_state_values(env, Q, alpha)
    print("Final Q-table:\n", Q)