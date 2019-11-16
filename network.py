import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

conv_layers = [
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(rate=0.5),

    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(rate=0.5),

    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Dropout(rate=0.5),

    layers.Conv2D(1024, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
]

fc_layers = Sequential([
    layers.Dense(1024, activation=tf.nn.relu),
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(62*4, activation=None),
])
