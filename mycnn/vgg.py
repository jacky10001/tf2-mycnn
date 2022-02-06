# -*- coding: utf-8 -*-
""" VGG Module
參考 VGG 的論文來實作模型架構

使用 Sub classing model 建立 VGG Block
加入 Batch Normalization
加入 Dropout

Refer:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition]
  (https://arxiv.org/abs/1409.1556) (ICLR 2015)
- [Keras API reference / Keras Applications / VGG16 and VGG19]
  (https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py)
- [vgg-nets | PyTorch]
  (https://pytorch.org/hub/pytorch_vision_vgg/)
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from .core.base_model import KerasModel


class ConvBN(models.Model):
    def __init__(self,
                 filters,
                 kernel_size=(3,3),
                 padding="same",
                 block_name=None,
                 times_name="1",
                 **kwargs) -> None:
        super().__init__(name=block_name + "_" + times_name, **kwargs)
        conv_name, bn_name, bn_name = None, None, None
        if block_name:
            conv_name = block_name + "_conv" + times_name
            bn_name = block_name + "_bn" + times_name
            act_name = block_name + "_act" + times_name
        self.conv = layers.Conv2D(filters, kernel_size, padding=padding, name=conv_name)
        self.bn = layers.BatchNormalization(name=bn_name)
        self.act = layers.ReLU(name=act_name)
    
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class VGG11(KerasModel):
    """ VGG11 (Type-A) """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # block 1
        block_name = "block1"
        filters = 64
        x = ConvBN(filters, block_name=block_name, times_name="1")(x_in)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 2
        block_name = "block2"
        filters = 128
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 3
        block_name = "block3"
        filters = 256
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 4
        block_name = "block4"
        filters = 512
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 5
        block_name = "block5"
        filters = 512
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

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
    """ VGG11 (Type-B) """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # block 1
        block_name = "block1"
        filters = 64
        x = ConvBN(filters, block_name=block_name, times_name="1")(x_in)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 2
        block_name = "block2"
        filters = 128
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 3
        block_name = "block3"
        filters = 256
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 4
        block_name = "block4"
        filters = 512
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 5
        block_name = "block5"
        filters = 512
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

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
    """ VGG16 (Type-D) """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # block 1
        block_name = "block1"
        filters = 64
        x = ConvBN(filters, block_name=block_name, times_name="1")(x_in)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 2
        block_name = "block2"
        filters = 128
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 3
        block_name = "block3"
        filters = 256
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = ConvBN(filters, block_name=block_name, times_name="3")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 4
        block_name = "block4"
        filters = 512
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = ConvBN(filters, block_name=block_name, times_name="3")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 5
        block_name = "block5"
        filters = 512
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = ConvBN(filters, block_name=block_name, times_name="3")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

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
    """ VGG19 (Type-E) """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # block 1
        block_name = "block1"
        filters = 64
        x = ConvBN(filters, block_name=block_name, times_name="1")(x_in)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 2
        block_name = "block2"
        filters = 128
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 3
        block_name = "block3"
        filters = 256
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = ConvBN(filters, block_name=block_name, times_name="3")(x)
        x = ConvBN(filters, block_name=block_name, times_name="4")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 4
        block_name = "block4"
        filters = 512
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = ConvBN(filters, block_name=block_name, times_name="3")(x)
        x = ConvBN(filters, block_name=block_name, times_name="4")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

        # block 5
        block_name = "block5"
        filters = 512
        x = ConvBN(filters, block_name=block_name, times_name="1")(x)
        x = ConvBN(filters, block_name=block_name, times_name="2")(x)
        x = ConvBN(filters, block_name=block_name, times_name="3")(x)
        x = ConvBN(filters, block_name=block_name, times_name="4")(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name=block_name+"_pool")(x)

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