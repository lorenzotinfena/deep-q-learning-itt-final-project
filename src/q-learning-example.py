#%%
import numpy as np
import gym
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from copy import deepcopy
class QAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        # inizializzo la Q-table
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
    def _optimal_policy(self, current_state): return np.argmax(self.Q[current_state])
    def _policy(self, current_state, epsilon):
        if np.random.rand() < epsilon: return self.env.action_space.sample()
        else: return self._optimal_policy(current_state)
    def start_episode(self, discount_factor, learning_rate, epsilon):
        # inizializzo le metriche
        total_reward = 0
        steps = 0
        # ottengo il primo state
        current_state = self.env.reset()
        # quando done è True l'episodio finisce
        done = False
        while not done:
            # policy non migliore locale (algoritmo epsilon-greedy)
            action = self._policy(current_state, epsilon)
            # compio la azione
            next_state, reward, done, _ = self.env.step(action)
            # aggiorno la Q-table
            if done: self.Q[current_state, action] += learning_rate * (reward - self.Q[current_state, action])
            else: self.Q[current_state, action] += learning_rate * (reward + discount_factor * np.max(self.Q[next_state]) - self.Q[current_state, action])
            # aggiorno le metriche
            total_reward += reward
            steps += 1
            # aggiorno il current_state
            current_state = next_state
        return total_reward, steps
#%%
TRAIN_EPISODES = 2500
TEST_EPISODES = 1000
def main():
    agent = QAgent(env=gym.make('Taxi-v3'))
    X = []
    Y_p = [] # total_reward ottenuti seguendo la policy non migliore (quella per il training)
    Y_opt_p = [] # total_reward ottenuti seguendo la policy migliore
    for episode in range(TRAIN_EPISODES):
        X.append(episode)
        total_reward, _ = agent.start_episode(discount_factor=0.99, learning_rate=0.1, epsilon=0.5)
        Y_p.append(total_reward)
        Q = deepcopy(agent.Q)
        total_reward, _ = agent.start_episode(discount_factor=0.99, learning_rate=0.1, epsilon=0)
        Y_opt_p.append(total_reward)
        agent.Q = Q
    X = np.array(X).reshape(-1, 100).mean(axis=1)
    Y_p = np.array(Y_p).reshape(-1, 100).mean(axis=1)
    Y_opt_p = np.array(Y_opt_p).reshape(-1, 100).mean(axis=1)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=X, y=Y_p, mode='lines', name='con policy migliore'), row=1, col=1)
    fig.add_trace(go.Scatter(x=X, y=Y_opt_p, mode='lines', name='con policy reale'), row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_yaxes(title_text="Total reward", row=1, col=1)
    fig.update_layout(height=500, width=750)
    fig.show()
    X = []
    Y = []
    for episode in range(TEST_EPISODES):
        total_reward, steps = agent.start_episode(discount_factor=0.99, learning_rate=0.1, epsilon=0)
        X.append(episode)
        Y.append(total_reward)
    print(f'media total reward: {np.mean(Y)}')
    print(f'deviazione standard total reward: {np.std(Y)}')
    fig = px.histogram(x=Y)
    fig.update_layout(height=300, width=400, showlegend=False)
    fig.update_xaxes(title_text="Reward")
    fig.update_yaxes(title_text="Count")
    fig.show()
if __name__ == '__main__':
    main()