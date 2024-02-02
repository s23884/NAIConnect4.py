"""
Opis problemu:
    Implementacja agenta DQN, który uczy się grać w grę "Gopher" na Atari, wykorzystując środowisko Gymnasium

Autorzy: Wiktor Krieger & Sebastian Augustyniak

Instrukcja użycia:
    1. Upewnij się, że masz zainstalowane wszystkie wymagane biblioteki: gymnasium, numpy, tensorflow.
    2. Skopiuj i wklej ten skrypt do swojego środowiska programistycznego.
    3. Uruchom skrypt.
    4. Obserwuj proces uczenia się agenta, który będzie wyświetlany w konsoli.
    5. W obecnej formie program działa tylko dla jednego epizodu, nie udalo się naprawić błędu
"""

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# Inicjalizacja środowiska Gymnasium dla gry Gopher na Atari
env = gym.make("ALE/Gopher-v5", render_mode='human')


def create_model():
    """
    Tworzy model sieci neuronowej dla agenta DQN.

    Returns:
        model: Zwraca skompilowany model sieci neuronowej.
    """
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(210, 160, 3)),
        layers.Conv2D(32, 8, strides=4, activation='relu'),
        layers.Conv2D(64, 4, strides=2, activation='relu'),
        layers.Conv2D(64, 3, strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(env.action_space.n, activation='linear')
    ])
    return model


class DQNAgent:
    def __init__(self, state_space, action_space):
        """
        Inicjalizuje agenta DQN.

        Args:
            state_space (tuple): Kształt przestrzeni stanów środowiska.
            action_space: Przestrzeń akcji środowiska.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = create_model()

    def remember(self, state, action, reward, next_state, done):
        """
        Zapamiętuje doświadczenie agenta.

        Args:
            state (np.array): Stan środowiska przed wykonaniem akcji.
            action (int): Wykonana akcja.
            reward (float): Otrzymana nagroda.
            next_state (np.array): Stan środowiska po wykonaniu akcji.
            done (bool): Czy epizod się zakończył.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Wybiera akcję dla danego stanu.

        Args:
            state (np.array): Stan środowiska.

        Returns:
            action (int): Wybrana akcja.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Trenuje model na podstawie doświadczeń z pamięci.

        Args:
            batch_size (int): Rozmiar próbki doświadczeń użytych do treningu.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)

            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_dqn(episodes):
    """
    Przeprowadza proces treningu agenta DQN.

    Args:
        episodes (int): Liczba epizodów treningowych.
    """
    loss = []
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    for e in range(episodes):
        full_state = env.reset()
        state = full_state[0]
        score = 0
        max_steps = 10000

        for i in range(max_steps):
            state = np.expand_dims(state, axis=0)
            action = agent.act(state)
            temp = env.step(action)
            print(temp)

            full_next_state, reward, done, info = temp[:4]
            next_state = full_next_state[0]
            next_state = np.expand_dims(next_state, axis=0)

            agent.remember(state, action, reward, next_state, done)
            state = next_state[0]
            score += reward

            if done:
                print(f"Episode: {e + 1}/{episodes}, score: {score}")
                break

        agent.replay(32)  # Rozmiar próbki doświadczeń użytych do treningu


if __name__ == "__main__":
    episodes = 100
    agent = DQNAgent(env.observation_space.shape, env.action_space)
    train_dqn(episodes)
