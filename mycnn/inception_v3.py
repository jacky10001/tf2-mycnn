# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from .core.base_model import KerasModel


class InceptionA(models.Model):
    def __init__(self,
                 c1x1: int,
                 c5x5red: int,
                 c5x5: int,
                 c3x3red: int,
                 c3x3_1: int,
                 c3x3_2: int,
                 p1x1: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv1x1 = layers.Conv2D(c1x1, (1,1), padding="same")
        self.a1_conv1x1_bn = layers.BatchNormalization()

        self.b1_conv1x1 = layers.Conv2D(c5x5red, (1,1), padding="same")
        self.b1_conv1x1_bn = layers.BatchNormalization()
        self.b2_conv5x5 = layers.Conv2D(c5x5, (5,5), padding="same")
        self.b2_conv5x5_bn = layers.BatchNormalization()

        self.c1_conv1x1 = layers.Conv2D(c3x3red, (1,1), padding="same")
        self.c1_conv1x1_bn = layers.BatchNormalization()
        self.c2_conv3x3 = layers.Conv2D(c3x3_1, (3,3), padding="same")
        self.c2_conv3x3_bn = layers.BatchNormalization()
        self.c3_conv3x3 = layers.Conv2D(c3x3_2, (3,3), padding="same")
        self.c3_conv3x3_bn = layers.BatchNormalization()

        self.d1_conv1x1 = layers.Conv2D(p1x1, (1,1), padding="same")
        self.d1_conv1x1_bn = layers.BatchNormalization()
    
    def call(self, x):
        x1 = self.a1_conv1x1(x)
        x1 = self.a1_conv1x1_bn(x1)
        x1 = layers.ReLU()(x1)
        
        x2 = self.b1_conv1x1(x)
        x2 = self.b1_conv1x1_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b2_conv5x5(x2)
        x2 = self.b2_conv5x5_bn(x2)
        x2 = layers.ReLU()(x2)
        
        x3 = self.c1_conv1x1(x)
        x3 = self.c1_conv1x1_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c2_conv3x3(x3)
        x3 = self.c2_conv3x3_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c3_conv3x3(x3)
        x3 = self.c3_conv3x3_bn(x3)
        x3 = layers.ReLU()(x3)

        x4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        x4 = self.d1_conv1x1(x4)
        x4 = self.d1_conv1x1_bn(x4)
        x4 = layers.ReLU()(x4)

        return layers.Concatenate()([x1,x2,x3,x4])


class InceptionV3(KerasModel):
    """
    Inception V3

    Note:
    論文有提到輔助分類器，這邊暫時不使用，因為訓練初期並不會有太大影響
    需要訓練到後期要更進一步提升準確度時，再考慮是否需要加入輔助分類器
    """
    def __init__(self,
                 input_shape=(299, 299, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # 299x299x3 -> 149x149x32
        x = layers.Conv2D(32, (3,3), strides=2, name="conv1_conv")(x_in)
        x = layers.BatchNormalization(name="conv1_bn")(x)
        x = layers.ReLU(name="conv1_relu")(x)

        # 149x149x32 -> 147x147x32
        x = layers.Conv2D(32, (3,3), name="conv2_conv")(x)
        x = layers.BatchNormalization(name="conv2_bn")(x)
        x = layers.ReLU(name="conv2_relu")(x)
        
        # 147x147x32 -> 147x147x64
        x = layers.Conv2D(64, (3,3), padding="same", name="conv3_conv")(x)
        x = layers.BatchNormalization(name="conv3_bn")(x)
        x = layers.ReLU(name="conv3_relu")(x)
        
        # 147x147x64 -> 73x73x64
        x = layers.MaxPooling2D((3,3), strides=(2,2), name="pool1")(x)

        # 73x73x64 -> 71x71x64
        x = layers.Conv2D(64, (3,3), name="conv4_conv")(x)
        x = layers.BatchNormalization(name="conv4_bn")(x)
        x = layers.ReLU(name="conv4_relu")(x)

        # 71x71x64 -> 35x35x80
        x = layers.Conv2D(80, (3,3), strides=2, name="conv5_conv")(x)
        x = layers.BatchNormalization(name="conv5_bn")(x)
        x = layers.ReLU(name="conv5_relu")(x)

        # 35x35x80 -> 35x35x192
        x = layers.Conv2D(192, (3,3), name="conv6_conv")(x)
        x = layers.BatchNormalization(name="conv6_bn")(x)
        x = layers.ReLU(name="conv6_relu")(x)

        # x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        # x = layers.Dropout(0.4)(x)
        # x = layers.Dense(self.classes_num, activation="linear", name="linear")(x)
        # x_out = layers.Dense(self.classes_num, activation="softmax", name="softmax")(x)
        
        self.setup_model(x_in, x, name="InceptionV3")
