# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from .core import KerasModel


class LeNet5(KerasModel):
    """ LeNet5 (超參數依照論文設置) """

    def __init__(self,
                 input_shape=(32, 32, 1),
                 classes_num=10,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
    
    def build(self, **kwargs):
        x_in = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='valid',
            activation='tanh'
        )(x_in)
        x = layers.AveragePooling2D((2,2))(x)

        x = layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='valid',
            activation='tanh')(x)
        x = layers.AveragePooling2D((2,2))(x)

        x = layers.Flatten()(x)

        x = layers.Dense(120, activation='tanh')(x)
        x = layers.Dense(84, activation='tanh')(x)
        x_out = layers.Dense(self.classes_num, activation='softmax')(x)
        
        self.setup_model(x_in, x_out, name="LeNet5", **kwargs)