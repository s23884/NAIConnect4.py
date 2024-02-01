import gym
import numpy as np
from agents.dqn_agent import DQNAgent


def preprocess_state(state):
    # Add any preprocessing here. For simplicity, this is just a placeholder.
    return state


def main():
    env = gym.make('Gopher-v0')
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size)

    episodes = 100
    for e in range(episodes):
        state = preprocess_state(env.reset())
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}")


if __name__ == '__main__':
    main()
