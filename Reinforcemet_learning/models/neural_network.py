import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import models, layers


def create_q_network(input_shape, action_size):
    model = models.Sequential([
        layers.Conv2D(8, (2, 2), strides=(1, 1), activation='relu', input_shape=input_shape),
        layers.Conv2D(16, (4, 4), strides=(2, 2), activation='relu'),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    return model
