# -*- coding: utf-8 -*-
"""
create complex classifier
    * using tf custom layer method

@author: Jacky Gao
@date: 2020/11/30
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


class Block(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(Block, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding='same',
                                           # use_bias=True,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=None)
        self.bn = tf.keras.layers.BatchNormalization()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
        })
        return config

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


def build_model(input_shape=(256,256,1)):
    # Amplitude
    xa = layers.Input(shape=input_shape, name='xa')
    x = Block(8)(xa)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = Block(16)(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = Block(32)(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = Block(64)(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    ya = layers.Dense(6, activation='softmax', name='ya')(x)
    
    # phase
    xp = layers.Input(shape=input_shape, name='xp')
    x = Block(8)(xp)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = Block(16)(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = Block(32)(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = Block(64)(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    yp = layers.Dense(6, activation='softmax', name='yp')(x)
    
    # complex
    x = layers.Concatenate(axis=-1)([ya,yp])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    yc = layers.Dense(6, activation='softmax', name='yc')(x)
    return models.Model(inputs={'xa': xa, 'xp': xp},
                        outputs={'ya': ya, 'yp': yp, 'yc': yc})


if __name__ == '__main__':    
    model = build_model()
    model.summary()