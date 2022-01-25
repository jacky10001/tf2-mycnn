# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from .core.base_model import KerasModel


class InceptionA(models.Model):
    def __init__(self,
                 c1x1: list,
                 c5x5: list,
                 c3x3: list,
                 p1x1: list,
                 use_bias: bool = False,
                 scale: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv1x1 = layers.Conv2D(c1x1[0], (1,1), padding="same", use_bias=use_bias)
        self.a1_conv1x1_bn = layers.BatchNormalization(scale=scale)
        self.a1_conv1x1_act = layers.ReLU()

        self.b1_conv5x5red = layers.Conv2D(c5x5[0], (1,1), padding="same", use_bias=use_bias)
        self.b1_conv5x5red_bn = layers.BatchNormalization(scale=scale)
        self.b1_conv1x1_act = layers.ReLU()
        self.b2_conv5x5 = layers.Conv2D(c5x5[1], (5,5), padding="same", use_bias=use_bias)
        self.b2_conv5x5_bn = layers.BatchNormalization(scale=scale)
        self.b2_conv1x1_act = layers.ReLU()

        self.c1_conv3x3red = layers.Conv2D(c3x3[0], (1,1), padding="same", use_bias=use_bias)
        self.c1_conv3x3red_bn = layers.BatchNormalization(scale=scale)
        self.c1_conv1x1_act = layers.ReLU()
        self.c2_conv3x3 = layers.Conv2D(c3x3[1], (3,3), padding="same", use_bias=use_bias)
        self.c2_conv3x3_bn = layers.BatchNormalization(scale=scale)
        self.c2_conv1x1_act = layers.ReLU()
        self.c3_conv3x3 = layers.Conv2D(c3x3[2], (3,3), padding="same", use_bias=use_bias)
        self.c3_conv3x3_bn = layers.BatchNormalization(scale=scale)
        self.c3_conv1x1_act = layers.ReLU()

        self.d1_pool = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')
        self.d2_conv1x1 = layers.Conv2D(p1x1[0], (1,1), padding="same", use_bias=use_bias)
        self.d2_conv1x1_bn = layers.BatchNormalization(scale=scale)
        self.d2_conv1x1_act = layers.ReLU()

        self.mix = layers.Concatenate()
    
    def call(self, x):
        x1 = self.a1_conv1x1(x)
        x1 = self.a1_conv1x1_bn(x1)
        x1 = self.a1_conv1x1_act(x1)
        
        x2 = self.b1_conv5x5red(x)
        x2 = self.b1_conv5x5red_bn(x2)
        x2 = self.b1_conv1x1_act(x2)
        x2 = self.b2_conv5x5(x2)
        x2 = self.b2_conv5x5_bn(x2)
        x2 = self.b2_conv1x1_act(x2)
        
        x3 = self.c1_conv3x3red(x)
        x3 = self.c1_conv3x3red_bn(x3)
        x3 = self.c1_conv1x1_act(x3)
        x3 = self.c2_conv3x3(x3)
        x3 = self.c2_conv3x3_bn(x3)
        x3 = self.c2_conv1x1_act(x3)
        x3 = self.c3_conv3x3(x3)
        x3 = self.c3_conv3x3_bn(x3)
        x3 = self.c3_conv1x1_act(x3)

        x4 = self.d1_pool(x)
        x4 = self.d2_conv1x1(x4)
        x4 = self.d2_conv1x1_bn(x4)
        x4 = self.d2_conv1x1_act(x4)

        return self.mix([x1,x2,x3,x4])


class InceptionB(models.Model):
    def __init__(self,
                 a_c3x3: list,
                 b_c3x3: list,
                 use_bias: bool = False,
                 scale: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv3x3 = layers.Conv2D(a_c3x3[0], (3,3), strides=(2, 2), padding="valid", use_bias=use_bias)
        self.a1_conv3x3_bn = layers.BatchNormalization(scale=scale)
        self.a1_conv3x3_act = layers.ReLU()

        self.b1_conv3x3red = layers.Conv2D(b_c3x3[0], (1,1), padding="same", use_bias=use_bias)
        self.b1_conv3x3red_bn = layers.BatchNormalization(scale=scale)
        self.b1_conv3x3red_act = layers.ReLU()
        self.b2_conv3x3 = layers.Conv2D(b_c3x3[1], (3,3), padding="same", use_bias=use_bias)
        self.b2_conv3x3_bn = layers.BatchNormalization(scale=scale)
        self.b2_conv3x3_act = layers.ReLU()
        self.b3_conv3x3 = layers.Conv2D(b_c3x3[2], (3,3), strides=(2,2), padding="valid", use_bias=use_bias)
        self.b3_conv3x3_bn = layers.BatchNormalization(scale=scale)
        self.b3_conv3x3_act = layers.ReLU()

        self.c1_pool = layers.MaxPooling2D((3,3), strides=(2,2))

        self.mix = layers.Concatenate()
    
    def call(self, x):
        x1 = self.a1_conv3x3(x)
        x1 = self.a1_conv3x3_bn(x1)
        x1 = self.a1_conv3x3_act(x1)
        
        x2 = self.b1_conv3x3red(x)
        x2 = self.b1_conv3x3red_bn(x2)
        x2 = self.b1_conv3x3red_act(x2)
        x2 = self.b2_conv3x3(x2)
        x2 = self.b2_conv3x3_bn(x2)
        x2 = self.b2_conv3x3_act(x2)
        x2 = self.b3_conv3x3(x2)
        x2 = self.b3_conv3x3_bn(x2)
        x2 = self.b3_conv3x3_act(x2)

        x3 = self.c1_pool(x)

        return self.mix([x1,x2,x3])


class InceptionC(models.Model):
    def __init__(self,
                 a_c1x1: list,
                 b_c7x7: list,
                 c_c7x7: list,
                 d_p1x1: list,
                 use_bias: bool = False,
                 scale: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv1x1 = layers.Conv2D(a_c1x1[0], (1,1), padding="same", use_bias=use_bias)
        self.a1_conv1x1_bn = layers.BatchNormalization(scale=scale)

        self.b1_conv7x7red = layers.Conv2D(b_c7x7[0], (1,1), padding="same", use_bias=use_bias)
        self.b1_conv7x7red_bn = layers.BatchNormalization(scale=scale)
        self.b1_conv7x7red_act = layers.ReLU()
        self.b2_conv1x7 = layers.Conv2D(b_c7x7[1], (1,7), padding="same", use_bias=use_bias)
        self.b2_conv1x7_bn = layers.BatchNormalization(scale=scale)
        self.b2_conv1x7_act = layers.ReLU()
        self.b3_conv7x1 = layers.Conv2D(b_c7x7[2], (7,1), padding="same", use_bias=use_bias)
        self.b3_conv7x1_bn = layers.BatchNormalization(scale=scale)
        self.b3_conv7x1_act = layers.ReLU()

        self.c1_conv7x7red = layers.Conv2D(c_c7x7[0], (1,1), padding="same", use_bias=use_bias)
        self.c1_conv7x7red_bn = layers.BatchNormalization(scale=scale)
        self.c1_conv7x7red_act = layers.ReLU()
        self.c2_conv7x1 = layers.Conv2D(c_c7x7[1], (7,1), padding="same", use_bias=use_bias)
        self.c2_conv7x1_bn = layers.BatchNormalization(scale=scale)
        self.c2_conv7x1_act = layers.ReLU()
        self.c3_conv1x7 = layers.Conv2D(c_c7x7[2], (1,7), padding="same", use_bias=use_bias)
        self.c3_conv1x7_bn = layers.BatchNormalization(scale=scale)
        self.c3_conv1x7_act = layers.ReLU()
        self.c4_conv7x1 = layers.Conv2D(c_c7x7[3], (7,1), padding="same", use_bias=use_bias)
        self.c4_conv7x1_bn = layers.BatchNormalization(scale=scale)
        self.c4_conv7x1_act = layers.ReLU()
        self.c5_conv1x7 = layers.Conv2D(c_c7x7[4], (1,7), padding="same", use_bias=use_bias)
        self.c5_conv1x7_bn = layers.BatchNormalization(scale=scale)
        self.c5_conv1x7_act = layers.ReLU()

        self.d1_pool = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')
        self.d2_conv1x1 = layers.Conv2D(d_p1x1[0], (1,1), padding="same", use_bias=use_bias)
        self.d2_conv1x1_bn = layers.BatchNormalization(scale=scale)
        self.d2_conv1x1_act = layers.ReLU()
        
        self.mix = layers.Concatenate()
    
    def call(self, x):
        x1 = self.a1_conv1x1(x)
        x1 = self.a1_conv1x1_bn(x1)
        x1 = self.c5_conv1x7_act(x1)
        
        x2 = self.b1_conv7x7red(x)
        x2 = self.b1_conv7x7red_bn(x2)
        x2 = self.b1_conv7x7red_act(x2)
        x2 = self.b2_conv1x7(x2)
        x2 = self.b2_conv1x7_bn(x2)
        x2 = self.b2_conv1x7_act(x2)
        x2 = self.b3_conv7x1(x2)
        x2 = self.b3_conv7x1_bn(x2)
        x2 = self.b3_conv7x1_act(x2)
        
        x3 = self.c1_conv7x7red(x)
        x3 = self.c1_conv7x7red_bn(x3)
        x3 = self.c1_conv7x7red_act(x3)
        x3 = self.c2_conv7x1(x3)
        x3 = self.c2_conv7x1_bn(x3)
        x3 = self.c2_conv7x1_act(x3)
        x3 = self.c3_conv1x7(x3)
        x3 = self.c3_conv1x7_bn(x3)
        x3 = self.c3_conv1x7_act(x3)
        x3 = self.c4_conv7x1(x3)
        x3 = self.c4_conv7x1_bn(x3)
        x3 = self.c4_conv7x1_act(x3)
        x3 = self.c5_conv1x7(x3)
        x3 = self.c5_conv1x7_bn(x3)
        x3 = self.c5_conv1x7_act(x3)

        x4 = self.d1_pool(x)
        x4 = self.d2_conv1x1(x4)
        x4 = self.d2_conv1x1_bn(x4)
        x4 = self.d2_conv1x1_act(x4)

        return self.mix([x1,x2,x3,x4])


class InceptionD(models.Model):
    def __init__(self,
                 a_c3x3: list,
                 b_c7x7x3: list,
                 use_bias: bool = False,
                 scale: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv3x3red = layers.Conv2D(a_c3x3[0], (1,1), padding="same", use_bias=use_bias)
        self.a1_conv3x3red_bn = layers.BatchNormalization(scale=scale)
        self.a1_conv3x3red_act = layers.ReLU()
        self.a2_conv3x3 = layers.Conv2D(a_c3x3[1], (3,3), strides=(2,2), padding="valid", use_bias=use_bias)
        self.a2_conv3x3_bn = layers.BatchNormalization(scale=scale)
        self.a2_conv3x3_act = layers.ReLU()

        self.b1_conv7x7x3red = layers.Conv2D(b_c7x7x3[0], (1,1), padding="same", use_bias=use_bias)
        self.b1_conv7x7x3red_bn = layers.BatchNormalization(scale=scale)
        self.b1_conv7x7x3red_act = layers.ReLU()
        self.b2_conv1x7x3 = layers.Conv2D(b_c7x7x3[1], (1,7), padding="same", use_bias=use_bias)
        self.b2_conv1x7x3_bn = layers.BatchNormalization(scale=scale)
        self.b2_conv1x7x3_act = layers.ReLU()
        self.b3_conv7x1x3 = layers.Conv2D(b_c7x7x3[2], (7,1), padding="same", use_bias=use_bias)
        self.b3_conv7x1x3_bn = layers.BatchNormalization(scale=scale)
        self.b2_conv7x1x3_act = layers.ReLU()
        self.b4_conv7x7x3 = layers.Conv2D(b_c7x7x3[3], (3,3), strides=(2,2), padding="valid", use_bias=use_bias)
        self.b4_conv7x7x3_bn = layers.BatchNormalization(scale=scale)
        self.b4_conv7x7x3_act = layers.ReLU()
        
        self.c1_pool = layers.MaxPooling2D((3,3), strides=(2,2))

        self.mix = layers.Concatenate()
    
    def call(self, x):
        x1 = self.a1_conv3x3red(x)
        x1 = self.a1_conv3x3red_bn(x1)
        x1 = self.a1_conv3x3red_act(x1)
        x1 = self.a2_conv3x3(x1)
        x1 = self.a2_conv3x3_bn(x1)
        x1 = self.a2_conv3x3_act(x1)
        
        x2 = self.b1_conv7x7x3red(x)
        x2 = self.b1_conv7x7x3red_bn(x2)
        x2 = self.b1_conv7x7x3red_act(x2)
        x2 = self.b2_conv1x7x3(x2)
        x2 = self.b2_conv1x7x3_bn(x2)
        x2 = self.b2_conv1x7x3_act(x2)
        x2 = self.b3_conv7x1x3(x2)
        x2 = self.b3_conv7x1x3_bn(x2)
        x2 = self.b2_conv7x1x3_act(x2)
        x2 = self.b4_conv7x7x3(x2)
        x2 = self.b4_conv7x7x3_bn(x2)
        x2 = self.b4_conv7x7x3_act(x2)

        x3 = self.c1_pool(x)

        return self.mix([x1,x2,x3])


class InceptionE(models.Model):
    def __init__(self,
                 a_c1x1: list,
                 b_c3x3: list,
                 c_c3x3: list,
                 d_p1x1: list,
                 use_bias: bool = False,
                 scale: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv1x1 = layers.Conv2D(a_c1x1[0], (1,1), padding="same", use_bias=use_bias)
        self.a1_conv1x1_bn = layers.BatchNormalization(scale=scale)
        self.a1_conv1x1_act = layers.ReLU()

        self.b1_conv3x3red = layers.Conv2D(b_c3x3[0], (1,1), padding="same", use_bias=use_bias)
        self.b1_conv3x3red_bn = layers.BatchNormalization(scale=scale)
        self.b1_conv3x3red_act = layers.ReLU()
        self.b2_conv1x3 = layers.Conv2D(b_c3x3[1], (1,3), padding="same", use_bias=use_bias)
        self.b2_conv1x3_bn = layers.BatchNormalization(scale=scale)
        self.b2_conv1x3_act = layers.ReLU()
        self.b3_conv3x1 = layers.Conv2D(b_c3x3[2], (3,1), padding="same", use_bias=use_bias)
        self.b3_conv3x1_bn = layers.BatchNormalization(scale=scale)
        self.b3_conv3x1_act = layers.ReLU()
        self.mix1 = layers.Concatenate()

        self.c1_conv3x3red = layers.Conv2D(c_c3x3[0], (1,1), padding="same", use_bias=use_bias)
        self.c1_conv3x3red_bn = layers.BatchNormalization(scale=scale)
        self.c1_conv3x3red_act = layers.ReLU()
        self.c2_conv3x3 = layers.Conv2D(c_c3x3[1], (3,3), padding="same", use_bias=use_bias)
        self.c2_conv3x3_bn = layers.BatchNormalization(scale=scale)
        self.c2_conv3x3_act = layers.ReLU()
        self.c3_conv1x3 = layers.Conv2D(c_c3x3[2], (1,3), padding="same", use_bias=use_bias)
        self.c3_conv1x3_bn = layers.BatchNormalization(scale=scale)
        self.c3_conv1x3_act = layers.ReLU()
        self.c4_conv3x1 = layers.Conv2D(c_c3x3[3], (3,1), padding="same", use_bias=use_bias)
        self.c4_conv3x1_bn = layers.BatchNormalization(scale=scale)
        self.c4_conv3x1_act = layers.ReLU()
        self.mix2 = layers.Concatenate()

        self.d1_pool = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')
        self.d2_conv1x1 = layers.Conv2D(d_p1x1[0], (1,1), padding="same", use_bias=use_bias)
        self.d2_conv1x1_bn = layers.BatchNormalization(scale=scale)
        self.d2_conv1x1_act = layers.ReLU()
        self.mix = layers.Concatenate()
    
    def call(self, x):
        x1 = self.a1_conv1x1(x)
        x1 = self.a1_conv1x1_bn(x1)
        x1 = self.a1_conv1x1_act(x1)
        
        x2 = self.b1_conv3x3red(x)
        x2 = self.b1_conv3x3red_bn(x2)
        x2 = self.b1_conv3x3red_act(x2)
        x2_1 = self.b2_conv1x3(x2)
        x2_1 = self.b2_conv1x3_bn(x2_1)
        x2_1 = self.b2_conv1x3_act(x2_1)
        x2_2 = self.b3_conv3x1(x2)
        x2_2 = self.b3_conv3x1_bn(x2_2)
        x2_2 = self.b3_conv3x1_act(x2_2)
        x2 = self.mix1([x2_1, x2_2])

        x3 = self.c1_conv3x3red(x)
        x3 = self.c1_conv3x3red_bn(x3)
        x3 = self.c1_conv3x3red_act(x3)
        x3 = self.c2_conv3x3(x3)
        x3 = self.c2_conv3x3_bn(x3)
        x3 = self.c2_conv3x3_act(x3)
        x3_1 = self.c3_conv1x3(x3)
        x3_1 = self.c3_conv1x3_bn(x3_1)
        x3_1 = self.c3_conv1x3_act(x3_1)
        x3_2 = self.c4_conv3x1(x3)
        x3_2 = self.c4_conv3x1_bn(x3_2)
        x3_2 = self.c4_conv3x1_act(x3_2)
        x3 = self.mix2([x3_1, x3_2])

        x4 = self.d1_pool(x)
        x4 = self.d2_conv1x1(x4)
        x4 = self.d2_conv1x1_bn(x4)
        x4 = self.d2_conv1x1_act(x4)

        return self.mix([x1,x2,x3,x4])


class InceptionV3(KerasModel):
    """
    Inception V3
    參考 Keras 及 PyTorch 的 Inception V3 API

    論文提到使用了三種 Inception Module，每種又實現不同 Inception Block，論文稱其為 Network in Network

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
        use_bias = False
        scale = False

        x_in = layers.Input(shape=self.input_shape, name="image")

        # 299x299x3
        x = layers.Conv2D(32, (3,3), strides=(2,2), padding="valid", use_bias=use_bias, name="conv1_conv")(x_in)
        x = layers.BatchNormalization(scale=scale, name="conv1_bn")(x)
        x = layers.ReLU(name="conv1_relu")(x)
        # 149x149x32
        x = layers.Conv2D(32, (3,3), padding="valid", use_bias=use_bias, name="conv2_conv")(x)
        x = layers.BatchNormalization(scale=scale, name="conv2_bn")(x)
        x = layers.ReLU(name="conv2_relu")(x)
        # 147x147x32
        x = layers.Conv2D(64, (3,3), padding="same", use_bias=use_bias, name="conv3_conv")(x)
        x = layers.BatchNormalization(scale=scale, name="conv3_bn")(x)
        x = layers.ReLU(name="conv3_relu")(x)
        # 147x147x64
        x = layers.MaxPooling2D((3,3), strides=(2,2), name="pool1")(x)
        # 73x73x64
        x = layers.Conv2D(80, (1,1), padding="valid", use_bias=use_bias, name="conv4_conv")(x)
        x = layers.BatchNormalization(scale=scale, name="conv4_bn")(x)
        x = layers.ReLU(name="conv4_relu")(x)
        # 73x73x80
        x = layers.Conv2D(192, (3,3), padding="valid", use_bias=use_bias, name="conv5_conv")(x)
        x = layers.BatchNormalization(scale=scale, name="conv5_bn")(x)
        x = layers.ReLU(name="conv5_relu")(x)
        # 71x71x192
        x = layers.MaxPooling2D((3,3), strides=(2,2), name="pool2")(x)
        # 35x35x192
        x = InceptionA([64], [48, 64], [64, 96, 96], [32], use_bias=use_bias, scale=scale, name="inception_a1")(x)
        # 35x35x256
        x = InceptionA([64], [48, 64], [64, 96, 96], [64], use_bias=use_bias, scale=scale, name="inception_a2")(x)
        # 35x35x288
        x = InceptionA([64], [48, 64], [64, 96, 96], [64], use_bias=use_bias, scale=scale, name="inception_a3")(x)
        # 35x35x288
        x = InceptionB([384], [64, 96, 96], name="inception_b1")(x)
        # 17x17x768
        x = InceptionC([192], [128, 128, 192], [128, 128, 128, 128, 192], [192], use_bias=use_bias, scale=scale, name="inception_c1")(x)
        # 17x17x768
        x = InceptionC([192], [160, 160, 192], [160, 160, 160, 160, 192], [192], use_bias=use_bias, scale=scale, name="inception_c2")(x)
        # 17x17x768
        x = InceptionC([192], [160, 160, 192], [160, 160, 160, 160, 192], [192], use_bias=use_bias, scale=scale, name="inception_c3")(x)
        # 17x17x768
        x = InceptionC([192], [192, 192, 192], [192, 192, 192, 192, 192], [192], use_bias=use_bias, scale=scale, name="inception_c4")(x)
        # 17x17x768
        x = InceptionD([192, 320], [192, 192, 192, 192], use_bias=use_bias, scale=scale, name="inception_d1")(x)
        # 8x8x1280
        x = InceptionE([320], [384, 384, 384], [448, 384, 384, 384], [192], use_bias=use_bias, scale=scale, name="inception_e1")(x)
        # 8x8x2048
        x = InceptionE([320], [384, 384, 384], [448, 384, 384, 384], [192], use_bias=use_bias, scale=scale, name="inception_e2")(x)
        # 8x8x2048

        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(self.classes_num, activation="linear", name="linear")(x)
        x_out = layers.Dense(self.classes_num, activation="softmax", name="softmax")(x)
        
        self.setup_model(x_in, x_out, name="InceptionV3")
