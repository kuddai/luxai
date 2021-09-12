import os

import tensorflow.keras as keras
import tensorflow as tf

from solution.utils import print


class Model(keras.Model):
    def __init__(self):
        super().__init__()

        self.map_embedding = keras.Sequential([
            keras.layers.Conv2D(8, 5, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2D(16, 5, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2D(32, 5, strides=2, padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128),
        ])

        self.game_vector_embeddings = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32),
        ])

        self.units_spatial = keras.Sequential([
            keras.layers.Conv2D(8, 3, strides=2, activation='relu'),
            keras.layers.Conv2D(16, 3, activation='relu'),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.Conv2D(32, 3),
        ])

        self.units_vector = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32),
        ])

        self.units_embedding = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64),
        ])

        self.advantages_head = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(5),
        ])
        self.state_value_head = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1),
        ])

        # self.units_mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=64, dropout=0.15)

    def call(self, inputs, training=None, mask=None):
        batch_shape = inputs[0].shape[0]

        game_map = self.map_embedding(inputs[0])
        game_vector = self.game_vector_embeddings(inputs[1], training=training)

        units_spatial = self.units_spatial(inputs[2])
        units_spatial = tf.reshape(units_spatial, (
            batch_shape,
            units_spatial.shape[-1],
        ))

        units_vector = self.units_vector(inputs[3])

        units_embedding = tf.concat((units_spatial, units_vector), -1)
        units_embedding = self.units_embedding(units_embedding)

        """
        units_mha = self.units_mha(
            units_embedding,
            units_embedding,
            attention_mask=tf.reshape(inputs[4], (batch_shape, 1, 1, inputs[4].shape[-1])),
            training=training
        )
        unit = units_mha[:, 0]"""
        unit = units_embedding

        state_vector = tf.concat((game_map, game_vector, unit), 1)

        advantages = self.advantages_head(state_vector, training=training)
        state_value = self.state_value_head(state_vector, training=training)

        q_values = state_value + advantages - tf.reduce_mean(advantages, -1, keepdims=True)

        return q_values
