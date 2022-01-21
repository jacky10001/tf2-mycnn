# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from .core.base_model import KerasModel


class InceptionV1(models.Model):
    def __init__(self,
                 c1x1: int,
                 c3x3red: int,
                 c3x3: int,
                 c5x5red: int,
                 c5x5: int,
                 p1x1: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.a_conv1x1 = layers.Conv2D(c1x1, (1,1), padding="same")
        self.a_conv1x1_bn = layers.BatchNormalization()
        self.a_conv1x1_act = layers.ReLU()

        self.b_conv1x1 = layers.Conv2D(c3x3red, (1,1), padding="same")
        self.b_conv1x1_bn = layers.BatchNormalization()
        self.b_conv1x1_act = layers.ReLU()
        self.b_conv3x3 = layers.Conv2D(c3x3, (3,3), padding="same")
        self.b_conv3x3_bn = layers.BatchNormalization()
        self.b_conv3x3_act = layers.ReLU()

        self.c_conv1x1 = layers.Conv2D(c5x5red, (1,1), padding="same")
        self.c_conv1x1_bn = layers.BatchNormalization()
        self.c_conv1x1_act = layers.ReLU()
        self.c_conv5x5 = layers.Conv2D(c5x5, (5,5), padding="same")
        self.c_conv5x5_bn = layers.BatchNormalization()
        self.c_conv5x5_act = layers.ReLU()

        self.d_pool = layers.MaxPooling2D((3, 3), strides=(1,1), padding="same")
        self.d_conv1x1 = layers.Conv2D(p1x1, (1,1), padding="same")
        self.d_conv1x1_bn = layers.BatchNormalization()
        self.d_conv1x1_act = layers.ReLU()
    
    def call(self, x):
        x1 = self.a_conv1x1(x)
        x1 = self.a_conv1x1_bn(x1)
        x1 = self.a_conv1x1_act(x1)
        
        x2 = self.b_conv1x1(x)
        x2 = self.b_conv1x1_bn(x2)
        x2 = self.b_conv1x1_act()(x2)
        x2 = self.b_conv3x3(x2)
        x2 = self.b_conv3x3_bn(x2)
        x2 = self.b_conv3x3_act(x2)
        
        x3 = self.c_conv1x1(x)
        x3 = self.c_conv1x1_bn(x3)
        x3 = self.c_conv1x1_act()(x3)
        x3 = self.c_conv5x5(x3)
        x3 = self.c_conv5x5_bn(x3)
        x3 = self.c_conv5x5_act(x3)

        x4 = self.d_pool(x)
        x4 = self.d_conv1x1(x4)
        x4 = self.d_conv1x1_bn(x4)
        x4 = self.d_conv1x1_act(x4)

        return layers.Concatenate()([x1,x2,x3,x4])


class GoogleNet(KerasModel):
    """
    GoogleNet (InceptionV1)

    Note:
    論文有提到輔助分類器，這邊暫時不使用，因為訓練初期並不會有太大影響
    需要訓練到後期要更進一步提升準確度時，再考慮是否需要加入輔助分類器
    """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs) -> None:
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # 224x224x3 -> 112x112x64
        x = layers.ZeroPadding2D(padding=(3,3), name='conv1_pad')(x_in)
        x = layers.Conv2D(64, (7,7), strides=(2,2), name='conv1_conv')(x)
        x = layers.BatchNormalization(name="conv1_bn")(x)
        x = layers.ReLU(name="conv1_relu")(x)
        x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3,3), strides=(2,2), name="pool1")(x)

        # 112x112x64 -> 56x56x192
        x = layers.Conv2D(64, (1,1), name='conv2_conv1')(x)
        x = layers.BatchNormalization(name="conv2_bn1")(x)
        x = layers.ReLU(name="conv2_relu1")(x)
        x = layers.Conv2D(192, (3,3), padding="same", name='conv2_conv2')(x)
        x = layers.BatchNormalization(name="conv2_bn2")(x)
        x = layers.ReLU(name="conv2_relu2")(x)
        x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name='pool2_pad')(x)
        x = layers.MaxPooling2D((3,3), strides=(2,2), name="pool2")(x)

        # 56x56x192 -> 14x14x480
        x = InceptionV1(64,96,128,16,32,32, name="inception_3a")(x)
        x = InceptionV1(128,128,192,32,96,64, name="inception_3b")(x)
        x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name='pool3_pad')(x)
        x = layers.MaxPooling2D((3,3), strides=(2,2), name="pool3")(x)

        # 14x14x480 -> 7x7x832
        x = InceptionV1(192,96,208,16,48,64, name="inception_4a")(x)
        x = InceptionV1(160,112,224,24,64,64, name="inception_4b")(x)
        x = InceptionV1(128,128,256,24,64,64, name="inception_4c")(x)
        x = InceptionV1(112,144,288,32,64,64, name="inception_4d")(x)
        x = InceptionV1(256,160,320,32,128,128, name="inception_4e")(x)
        x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name='pool4_pad')(x)
        x = layers.MaxPooling2D((3,3), strides=(2,2), name="pool4")(x)

        # 7x7x832 -> 1x1x1024
        x = InceptionV1(256,160,320,32,128,128, name="inception_5a")(x)
        x = InceptionV1(384,192,384,48,128,128, name="inception_5b")(x)

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(self.classes_num, activation='linear', name="linear")(x)
        x_out = layers.Dense(self.classes_num, activation='softmax', name="softmax")(x)
        
        self.setup_model(x_in, x_out, name="GoogLeNet")
