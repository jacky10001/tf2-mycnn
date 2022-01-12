# -*- coding: utf-8 -*-
""" VGG Module

Note:
- Convolutional Layer 都用上 BatchNormalization Layer
- Fully Connection Layer 加入 Dropout Layer 機制
"""

import tensorflow as tf
from tensorflow.keras import layers
from .core.base_model import KerasModel


class VGG11(KerasModel):
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        """ VGG11 (Type-A) """
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # Common parameter for layer
        padding = 'same'
        kernel_size = (3, 3)
        pool_size = (2, 2)

        # block 1
        block_name = "block1"
        filters = 64
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x_in)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 2
        block_name = "block2"
        filters = 128
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 3
        block_name = "block3"
        filters = 256
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 4
        block_name = "block4"
        filters = 512
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 5
        block_name = "block5"
        filters = 512
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        x = layers.Flatten(name="flatten")(x)

        block_name = "fc1"
        x = layers.Dense(4096, name=block_name)(x)
        x = layers.ReLU(name=block_name+"_relu")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)

        block_name = "fc2"
        x = layers.Dense(4096, name=block_name)(x)
        x = layers.ReLU(name=block_name+"_relu")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)

        x_out = layers.Dense(self.classes_num, activation='softmax', name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="VGG11")


class VGG13(KerasModel):
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        """ VGG11 (Type-B) """
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # Common parameter for layer
        padding = 'same'
        kernel_size = (3, 3)
        pool_size = (2, 2)

        # block 1
        block_name = "block1"
        filters = 64
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x_in)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 2
        block_name = "block2"
        filters = 128
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 3
        block_name = "block3"
        filters = 256
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 4
        block_name = "block4"
        filters = 512
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 5
        block_name = "block5"
        filters = 512
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        x = layers.Flatten(name="flatten")(x)

        block_name = "fc1"
        x = layers.Dense(4096, name=block_name)(x)
        x = layers.ReLU(name=block_name+"_relu")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)

        block_name = "fc2"
        x = layers.Dense(4096, name=block_name)(x)
        x = layers.ReLU(name=block_name+"_relu")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)

        x_out = layers.Dense(self.classes_num, activation='softmax', name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="VGG13")


class VGG16(KerasModel):
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        """ VGG16 (Type-D) """
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # Common parameter for layer
        padding = 'same'
        kernel_size = (3, 3)
        pool_size = (2, 2)

        # block 1
        block_name = "block1"
        filters = 64
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x_in)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 2
        block_name = "block2"
        filters = 128
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 3
        block_name = "block3"
        filters = 256
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv3")(x)
        x = layers.BatchNormalization(name=block_name+"_bn3")(x)
        x = layers.ReLU(name=block_name+"_relu3")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 4
        block_name = "block4"
        filters = 512
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv3")(x)
        x = layers.BatchNormalization(name=block_name+"_bn3")(x)
        x = layers.ReLU(name=block_name+"_relu3")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 5
        block_name = "block5"
        filters = 512
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv3")(x)
        x = layers.BatchNormalization(name=block_name+"_bn3")(x)
        x = layers.ReLU(name=block_name+"_relu3")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        x = layers.Flatten(name="flatten")(x)

        block_name = "fc1"
        x = layers.Dense(4096, name=block_name)(x)
        x = layers.ReLU(name=block_name+"_relu")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)

        block_name = "fc2"
        x = layers.Dense(4096, name=block_name)(x)
        x = layers.ReLU(name=block_name+"_relu")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)

        x_out = layers.Dense(self.classes_num, activation='softmax', name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="VGG16")


class VGG19(KerasModel):
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        """ VGG19 (Type-E) """
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # Common parameter for layer
        padding = 'same'
        kernel_size = (3, 3)
        pool_size = (2, 2)

        # block 1
        block_name = "block1"
        filters = 64
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x_in)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 2
        block_name = "block2"
        filters = 128
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 3
        block_name = "block3"
        filters = 256
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv3")(x)
        x = layers.BatchNormalization(name=block_name+"_bn3")(x)
        x = layers.ReLU(name=block_name+"_relu3")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv4")(x)
        x = layers.BatchNormalization(name=block_name+"_bn4")(x)
        x = layers.ReLU(name=block_name+"_relu4")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 4
        block_name = "block4"
        filters = 512
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv3")(x)
        x = layers.BatchNormalization(name=block_name+"_bn3")(x)
        x = layers.ReLU(name=block_name+"_relu3")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv4")(x)
        x = layers.BatchNormalization(name=block_name+"_bn4")(x)
        x = layers.ReLU(name=block_name+"_relu4")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        # block 5
        block_name = "block5"
        filters = 512
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv3")(x)
        x = layers.BatchNormalization(name=block_name+"_bn3")(x)
        x = layers.ReLU(name=block_name+"_relu3")(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding, name=block_name+"_conv4")(x)
        x = layers.BatchNormalization(name=block_name+"_bn4")(x)
        x = layers.ReLU(name=block_name+"_relu4")(x)
        x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

        x = layers.Flatten(name="flatten")(x)

        block_name = "fc1"
        x = layers.Dense(4096, name=block_name)(x)
        x = layers.ReLU(name=block_name+"_relu")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)

        block_name = "fc2"
        x = layers.Dense(4096, name=block_name)(x)
        x = layers.ReLU(name=block_name+"_relu")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)

        x_out = layers.Dense(self.classes_num, activation='softmax', name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="VGG19")