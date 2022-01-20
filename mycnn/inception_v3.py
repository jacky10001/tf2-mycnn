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
                 c3x3: int,
                 p1x1: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv1x1 = layers.Conv2D(c1x1, (1,1), padding="same")
        self.a1_conv1x1_bn = layers.BatchNormalization()

        self.b1_conv5x5red = layers.Conv2D(c5x5red, (1,1), padding="same")
        self.b1_conv5x5red_bn = layers.BatchNormalization()
        self.b2_conv5x5 = layers.Conv2D(c5x5, (5,5), padding="same")
        self.b2_conv5x5_bn = layers.BatchNormalization()

        self.c1_conv3x3red = layers.Conv2D(c3x3red, (1,1), padding="same")
        self.c1_conv3x3red_bn = layers.BatchNormalization()
        self.c2_conv3x3 = layers.Conv2D(c3x3, (3,3), padding="same")
        self.c2_conv3x3_bn = layers.BatchNormalization()
        self.c3_conv3x3 = layers.Conv2D(c3x3, (3,3), padding="same")
        self.c3_conv3x3_bn = layers.BatchNormalization()

        self.d1_conv1x1 = layers.Conv2D(p1x1, (1,1), padding="same")
        self.d1_conv1x1_bn = layers.BatchNormalization()
    
    def call(self, x):
        x1 = self.a1_conv1x1(x)
        x1 = self.a1_conv1x1_bn(x1)
        x1 = layers.ReLU()(x1)
        
        x2 = self.b1_conv5x5red(x)
        x2 = self.b1_conv5x5red_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b2_conv5x5(x2)
        x2 = self.b2_conv5x5_bn(x2)
        x2 = layers.ReLU()(x2)
        
        x3 = self.c1_conv3x3red(x)
        x3 = self.c1_conv3x3red_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c2_conv3x3(x3)
        x3 = self.c2_conv3x3_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c3_conv3x3(x3)
        x3 = self.c3_conv3x3_bn(x3)
        x3 = layers.ReLU()(x3)

        x4 = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
        x4 = self.d1_conv1x1(x4)
        x4 = self.d1_conv1x1_bn(x4)
        x4 = layers.ReLU()(x4)

        return layers.Concatenate()([x1,x2,x3,x4])


class InceptionB(models.Model):
    def __init__(self,
                 a_c3x3: int,
                 b_c3x3red: int,
                 b_c3x3: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv3x3 = layers.Conv2D(a_c3x3, (3,3), strides=(2, 2), padding="valid")
        self.a1_conv3x3_bn = layers.BatchNormalization()

        self.b1_conv3x3red = layers.Conv2D(b_c3x3red, (1,1), padding="same")
        self.b1_conv3x3red_bn = layers.BatchNormalization()
        self.b2_conv3x3 = layers.Conv2D(b_c3x3, (3,3), padding="same")
        self.b2_conv3x3_bn = layers.BatchNormalization()
        self.b3_conv3x3 = layers.Conv2D(b_c3x3, (3,3), strides=(2,2), padding="valid")
        self.b3_conv3x3_bn = layers.BatchNormalization()
    
    def call(self, x):
        x1 = self.a1_conv3x3(x)
        x1 = self.a1_conv3x3_bn(x1)
        x1 = layers.ReLU()(x1)
        
        x2 = self.b1_conv3x3red(x)
        x2 = self.b1_conv3x3red_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b2_conv3x3(x2)
        x2 = self.b2_conv3x3_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b3_conv3x3(x2)
        x2 = self.b3_conv3x3_bn(x2)
        x2 = layers.ReLU()(x2)

        x3 = layers.MaxPooling2D((3,3), strides=(2,2))(x)

        return layers.Concatenate()([x1,x2,x3])


class InceptionC(models.Model):
    def __init__(self,
                 a_c1x1: int,
                 b_c7x7red: int,
                 b_c7x7: int,
                 c_c7x7red: int,
                 c_c7x7: int,
                 d_p1x1: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv1x1 = layers.Conv2D(a_c1x1, (1,1), padding="same")
        self.a1_conv1x1_bn = layers.BatchNormalization()

        self.b1_conv7x7red = layers.Conv2D(b_c7x7red, (1,1), padding="same")
        self.b1_conv7x7red_bn = layers.BatchNormalization()
        self.b2_conv1x7 = layers.Conv2D(b_c7x7red, (1,7), padding="same")
        self.b2_conv1x7_bn = layers.BatchNormalization()
        self.b2_conv7x1 = layers.Conv2D(b_c7x7, (7,1), padding="same")
        self.b2_conv7x1_bn = layers.BatchNormalization()

        self.c1_conv7x7red = layers.Conv2D(c_c7x7red, (1,1), padding="same")
        self.c1_conv7x7red_bn = layers.BatchNormalization()
        self.c2_conv7x1 = layers.Conv2D(c_c7x7red, (7,1), padding="same")
        self.c2_conv7x1_bn = layers.BatchNormalization()
        self.c3_conv1x7 = layers.Conv2D(c_c7x7red, (1,7), padding="same")
        self.c3_conv1x7_bn = layers.BatchNormalization()
        self.c4_conv7x1 = layers.Conv2D(c_c7x7red, (7,1), padding="same")
        self.c4_conv7x1_bn = layers.BatchNormalization()
        self.c5_conv1x7 = layers.Conv2D(c_c7x7, (1,7), padding="same")
        self.c5_conv1x7_bn = layers.BatchNormalization()

        self.d1_conv1x1 = layers.Conv2D(d_p1x1, (1,1), padding="same")
        self.d1_conv1x1_bn = layers.BatchNormalization()
    
    def call(self, x):
        x1 = self.a1_conv1x1(x)
        x1 = self.a1_conv1x1_bn(x1)
        x1 = layers.ReLU()(x1)
        
        x2 = self.b1_conv7x7red(x)
        x2 = self.b1_conv7x7red_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b2_conv1x7(x2)
        x2 = self.b2_conv1x7_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b2_conv7x1(x2)
        x2 = self.b2_conv7x1_bn(x2)
        x2 = layers.ReLU()(x2)
        
        x3 = self.c1_conv7x7red(x)
        x3 = self.c1_conv7x7red_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c2_conv7x1(x3)
        x3 = self.c2_conv7x1_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c3_conv1x7(x3)
        x3 = self.c3_conv1x7_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c4_conv7x1(x3)
        x3 = self.c4_conv7x1_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c5_conv1x7(x3)
        x3 = self.c5_conv1x7_bn(x3)
        x3 = layers.ReLU()(x3)

        x4 = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
        x4 = self.d1_conv1x1(x4)
        x4 = self.d1_conv1x1_bn(x4)
        x4 = layers.ReLU()(x4)

        return layers.Concatenate()([x1,x2,x3,x4])


class InceptionD(models.Model):
    def __init__(self,
                 a_c3x3red: int,
                 a_c3x3: int,
                 b_c7x7x3red: int,
                 b_c7x7x3: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv3x3red = layers.Conv2D(a_c3x3red, (1,1), padding="same")
        self.a1_conv3x3red_bn = layers.BatchNormalization()
        self.a2_conv3x3 = layers.Conv2D(a_c3x3, (3,3), strides=(2,2), padding='valid')
        self.a2_conv3x3_bn = layers.BatchNormalization()

        self.b1_conv7x7x3 = layers.Conv2D(b_c7x7x3red, (1,1), padding="same")
        self.b1_conv7x7x3_bn = layers.BatchNormalization()
        self.b2_conv7x7x3 = layers.Conv2D(b_c7x7x3red, (1,7), padding="same")
        self.b2_conv7x7x3_bn = layers.BatchNormalization()
        self.b3_conv7x7x3 = layers.Conv2D(b_c7x7x3red, (7,1), padding="same")
        self.b3_conv7x7x3_bn = layers.BatchNormalization()
        self.b4_conv7x7x3 = layers.Conv2D(b_c7x7x3, (3,3), strides=(2,2), padding='valid')
        self.b4_conv7x7x3_bn = layers.BatchNormalization()
    
    def call(self, x):
        x1 = self.a1_conv3x3red(x)
        x1 = self.a1_conv3x3red_bn(x1)
        x1 = layers.ReLU()(x1)
        x1 = self.a2_conv3x3(x)
        x1 = self.a2_conv3x3_bn(x1)
        x1 = layers.ReLU()(x1)
        
        x2 = self.b1_conv7x7x3(x)
        x2 = self.b1_conv7x7x3_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b2_conv7x7x3(x2)
        x2 = self.b2_conv7x7x3_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b3_conv7x7x3(x2)
        x2 = self.b3_conv7x7x3_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b3_conv7x7x3(x2)
        x2 = self.b3_conv7x7x3_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b4_conv7x7x3(x2)
        x2 = self.b4_conv7x7x3_bn(x2)
        x2 = layers.ReLU()(x2)

        x3 = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        return layers.Concatenate()([x1,x2,x3])


class InceptionE(models.Model):
    def __init__(self,
                 a_c1x1: int,
                 b_c3x3red: int,
                 b_c3x3: int,
                 c_c3x3red: int,
                 c_c3x3: int,
                 d_p1x1: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a1_conv1x1 = layers.Conv2D(a_c1x1, (1,1), padding="same")
        self.a1_conv1x1_bn = layers.BatchNormalization()

        self.b1_conv7x7red = layers.Conv2D(b_c3x3red, (1,1), padding="same")
        self.b1_conv7x7red_bn = layers.BatchNormalization()
        self.b2_conv1x7 = layers.Conv2D(b_c3x3red, (1,3), padding="same")
        self.b2_conv1x7_bn = layers.BatchNormalization()
        self.b2_conv7x1 = layers.Conv2D(b_c3x3, (3,1), padding="same")
        self.b2_conv7x1_bn = layers.BatchNormalization()

        self.c1_conv7x7red = layers.Conv2D(c_c3x3red, (1,1), padding="same")
        self.c1_conv7x7red_bn = layers.BatchNormalization()
        self.c2_conv7x1 = layers.Conv2D(c_c3x3red, (3,1), padding="same")
        self.c2_conv7x1_bn = layers.BatchNormalization()
        self.c3_conv1x7 = layers.Conv2D(c_c3x3red, (1,3), padding="same")
        self.c3_conv1x7_bn = layers.BatchNormalization()
        self.c4_conv7x1 = layers.Conv2D(c_c3x3red, (3,1), padding="same")
        self.c4_conv7x1_bn = layers.BatchNormalization()
        self.c5_conv1x7 = layers.Conv2D(c_c3x3, (1,3), padding="same")
        self.c5_conv1x7_bn = layers.BatchNormalization()

        self.d1_conv1x1 = layers.Conv2D(d_p1x1, (1,1), padding="same")
        self.d1_conv1x1_bn = layers.BatchNormalization()
    
    def call(self, x):
        x1 = self.a1_conv1x1(x)
        x1 = self.a1_conv1x1_bn(x1)
        x1 = layers.ReLU()(x1)
        
        x2 = self.b1_conv7x7red(x)
        x2 = self.b1_conv7x7red_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b2_conv1x7(x2)
        x2 = self.b2_conv1x7_bn(x2)
        x2 = layers.ReLU()(x2)
        x2 = self.b2_conv7x1(x2)
        x2 = self.b2_conv7x1_bn(x2)
        x2 = layers.ReLU()(x2)
        
        x3 = self.c1_conv7x7red(x)
        x3 = self.c1_conv7x7red_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c2_conv7x1(x3)
        x3 = self.c2_conv7x1_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c3_conv1x7(x3)
        x3 = self.c3_conv1x7_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c4_conv7x1(x3)
        x3 = self.c4_conv7x1_bn(x3)
        x3 = layers.ReLU()(x3)
        x3 = self.c5_conv1x7(x3)
        x3 = self.c5_conv1x7_bn(x3)
        x3 = layers.ReLU()(x3)

        x4 = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
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
        x = layers.Conv2D(32, (3,3), strides=(2,2), name="conv1_conv")(x_in)
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

        # 73x73x64 -> 71x71x80
        x = layers.Conv2D(80, (1,1), name="conv4_conv")(x)
        x = layers.BatchNormalization(name="conv4_bn")(x)
        x = layers.ReLU(name="conv4_relu")(x)

        # 71x71x64 -> 35x35x192
        x = layers.Conv2D(192, (3,3), name="conv5_conv")(x)
        x = layers.BatchNormalization(name="conv5_bn")(x)
        x = layers.ReLU(name="conv5_relu")(x)

        # 35x35x80 -> 35x35x192
        x = layers.MaxPooling2D((3,3), strides=(2,2), name="pool2")(x)

        x = InceptionA(64, 48, 64, 64, 96, 32, name="inception_a1")(x)
        x = InceptionA(64, 48, 64, 64, 96, 64, name="inception_a2")(x)
        x = InceptionA(64, 48, 64, 64, 96, 64, name="inception_a3")(x)
        x = InceptionB(384, 64, 96, name="inception_b1")(x)
        x = InceptionC(192, 128, 192, 128, 192, 192, name="inception_c1")(x)
        x = InceptionC(192, 160, 192, 160, 192, 192, name="inception_c2")(x)
        x = InceptionC(192, 160, 192, 160, 192, 192, name="inception_c3")(x)
        x = InceptionC(192, 192, 192, 192, 192, 192, name="inception_c4")(x)
        x = InceptionD(192, 320, 192, 192, name="inception_d1")(x)

        # x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        # x = layers.Dropout(0.4)(x)
        # x = layers.Dense(self.classes_num, activation="linear", name="linear")(x)
        # x_out = layers.Dense(self.classes_num, activation="softmax", name="softmax")(x)
        
        self.setup_model(x_in, x, name="InceptionV3")
