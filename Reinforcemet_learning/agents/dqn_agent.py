import numpy as np
from models.neural_network import create_q_network
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=64,
                 update_every=4):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.learning_rate = learning_rate

        self.q_network = create_q_network(state_shape, action_size)
        self.target_q_network = create_q_network(state_shape, action_size)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.total_steps = 0

    def select_action(self, state, epsilon=0.01):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = self.q_network.predict(state[None, :], verbose=0)
            return np.argmax(q_values[0])

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        target_q_values = rewards + self.gamma * np.amax(self.target_q_network.predict(next_states, verbose=0),
                                                         axis=1) * (1 - dones)
        target_q_values_full = self.q_network.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_q_values_full[i, action] = target_q_values[i]

        self.q_network.fit(states, target_q_values_full, epochs=1, verbose=0)

        if self.total_steps % self.update_every == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

        self.total_steps += 1

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
