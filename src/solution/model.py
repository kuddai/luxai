import os

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

from solution.utils import print
import solution.constants as const


class ResnetLayer(keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()

        self.conv1 = keras.layers.Conv2D(out_channels, 3, activation='relu', padding='same')
        self.conv2 = keras.layers.Conv2D(out_channels, 3, padding='same')

    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()

        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, attention_axes=2)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class VisualTransformer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.linear = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64),
        ])
        self.positional_encoding = positional_encoding(25, 64)

        self.encoder = keras.Sequential([
            TransformerBlock(64, 4, 128),
            TransformerBlock(64, 4, 128),
        ])


    def call(self, inputs):
        assert inputs.shape[1:] == (15, 15, 6)

        batch_size = inputs.shape[0]

        patches = tf.image.extract_patches(inputs, (1, 3, 3, 1), (1, 3, 3, 1), (1, 1, 1, 1), 'VALID')
        flatten_patches = tf.reshape(patches, (batch_size, 25, 54))
        embeddings = self.linear(flatten_patches)
        embeddings += self.positional_encoding

        result = self.encoder(embeddings)
        unit_block = result[:, 13]
        return unit_block


class Model(keras.Model):
    def __init__(self):
        super().__init__()

        self.map_embedding = keras.Sequential([
            keras.layers.Conv2D(8, 5, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2D(16, 5, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2D(32, 5, strides=2, padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
        ])

        self.game_vector_embeddings = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
        ])

        self.units_spatial = keras.Sequential([
            keras.layers.Conv2D(16, 3, padding='same'),
            ResnetLayer(16),
            keras.layers.Conv2D(32, 3, strides=2, padding='same'),
            ResnetLayer(32),
            keras.layers.Conv2D(64, 3, strides=2, padding='same'),
            ResnetLayer(64),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
        ])
        """
        self.units_spatial = VisualTransformer()

        """
        self.units_vector = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
        ])

        self.units_embedding = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
        ])

        self.state_embedding = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
        ])

        self.advantages_head = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(7),
        ])
        self.state_value_head = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1),
        ])

        # self.units_mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=64, dropout=0.15)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        batch_size = inputs[0].shape[0]

        game_map = self.map_embedding(inputs[0])
        game_vector = self.game_vector_embeddings(inputs[1], training=training)
        game_vector = tf.concat((game_vector, game_map), -1)
        game_vector = tf.repeat(tf.expand_dims(game_vector, 1), const.MAX_UNITS, axis=1)

        units_spatial_input = tf.reshape(inputs[2], (-1, *inputs[2].shape[2:]))

        units_spatial = self.units_spatial(units_spatial_input)
        units_spatial = tf.reshape(units_spatial, (*inputs[2].shape[:2], -1))

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
        #unit = tf.where(tf.cast(inputs[4], tf.bool), unit, tf.zeros(unit.shape, unit.dtype))

        unit_state_vector = tf.concat((game_vector, units_embedding), -1)

        state_embedding = self.state_embedding(unit_state_vector)

        state_value = self.state_value_head(state_embedding, training=training)
        advantages = self.advantages_head(state_embedding, training=training)

        q_values = state_value + advantages - tf.reduce_mean(advantages, -1, keepdims=True)

        if const.DEBUG:
            units_mask = tf.expand_dims(inputs[4], 2)
            meaningful_state_values = tf.boolean_mask(state_value, units_mask)
            meaningful_q_values = tf.boolean_mask(q_values, tf.squeeze(units_mask), axis=1)
            avg_state_value = tf.reduce_mean(meaningful_state_values)
            tf.print('state value', avg_state_value)
            tf.print('q values', meaningful_q_values, summarize=-1)

        return q_values
