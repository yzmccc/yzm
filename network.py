import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

conv_layers = [
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(rate=0.6),

    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(rate=0.6),

    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(rate=0.6),

    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.tanh),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.tanh),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(rate=0.6),

    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.tanh),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.tanh),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Flatten()
]

fc_layers = Sequential([
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(62, activation=None),
])
