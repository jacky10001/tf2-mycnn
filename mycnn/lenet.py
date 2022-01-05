# -*- coding: utf-8 -*-

from tensorflow.keras import layers
from .common.core import KerasModel


class LeNet5(KerasModel):
    def __init__(self):
        super().__init__()
    
    def build(self, input_shape=(32, 32, 1), classes_num=10, **kwargs):
        x_in = layers.Input(shape=input_shape)

        x = layers.Conv2D(filters=6, 
                          kernel_size=(5, 5),
                          strides=(1, 1),
                          padding='valid',
                          activation='tanh')(x_in)
        x = layers.AveragePooling2D((2,2))(x)

        x = layers.Conv2D(filters=16, 
                          kernel_size=(5, 5),
                          strides=(1, 1),
                          padding='valid',
                          activation='tanh')(x)
        x = layers.AveragePooling2D((2,2))(x)

        x = layers.Flatten()(x)

        x = layers.Dense(120, activation='tanh')(x)
        x = layers.Dense(84, activation='tanh')(x)
        x_out = layers.Dense(classes_num, activation='softmax')(x)
        
        self.setup_model(x_in, x_out)
