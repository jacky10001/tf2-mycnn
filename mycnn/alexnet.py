# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from .core.base_model import KerasModel


class LRN(layers.Layer):
    """ Implement Local Response Normalization """
    def __init__(self,
                 alpha=0.0001,
                 k=2,
                 beta=0.75,
                 n=5,
                 **kwargs):
        super(LRN, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
    
    def call(self, x):
        return tf.nn.lrn(x,
                         depth_radius=self.n,
                         bias=self.k,
                         alpha=self.alpha,
                         beta=self.beta)

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AlexNet(KerasModel):
    """ AlexNet+BN (超參數依照論文設置) """

    def __init__(self,
                 input_shape=(227, 227, 3),
                 classes_num=10,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self, **kwargs):
        x_in = layers.Input(shape=self.input_shape, name="image")

        x = layers.Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            # kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            padding='valid'
        )(x_in)
        x = layers.BatchNormalization()(x)
        # x = LRN()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2))(x)

        x = layers.Conv2D(
            filters=256, 
            kernel_size=(5, 5),
            # kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            # bias_initializer='ones',
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        # x = LRN()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2))(x)

        x = layers.Conv2D(
            filters=384,
            kernel_size=(3, 3),
            # kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=384,
            kernel_size=(3, 3),
            # kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            # bias_initializer='ones',
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            # kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            # bias_initializer='ones',
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2)
        )(x)

        x = layers.Flatten()(x)

        x = layers.Dense(
            4096,
            #  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            # bias_initializer='ones'
        )(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(
            4096,
            # kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            # bias_initializer='ones'
        )(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x)

        x_out = layers.Dense(self.classes_num, activation='softmax')(x)
        
        self.setup_model(x_in, x_out, name="AlexNet", **kwargs)