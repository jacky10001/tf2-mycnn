# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from .core.base_model import KerasModel


class ResBlock(models.Model):
    def __init__(self, filters, strides=(1,1), conv_shortcut=False, **kwargs) -> None:
        self._conv_shortcut = conv_shortcut
        super().__init__(**kwargs)
        if conv_shortcut:
            self.conv_shortcut = layers.Conv2D(4*filters, (1,1), strides=strides)
            self.bn_shortcut = layers.BatchNormalization()
        
        self.conv1 = layers.Conv2D(filters, (1,1), strides=strides)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters, (3,3), padding="same")
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()

        self.conv3 = layers.Conv2D(4*filters, (1,1))
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.ReLU()

        self.add = layers.Add()
    
    def call(self, inputs):
        if self._conv_shortcut:
            x_shortcut = self.conv_shortcut(inputs)
            x_shortcut = self.bn_shortcut(x_shortcut)
        else:
            x_shortcut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.add([x,x_shortcut])
        x = self.act3(x)
        return x


class ResNet18(KerasModel):
    """ ResNet18 """
    
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self, **kwargs):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # input stem
        x = layers.ZeroPadding2D(padding=(3,3), name='conv1_pad')(x_in)
        x = layers.Conv2D(64, (7,7), strides=2, name='conv1_conv')(x)
        x = layers.BatchNormalization(name="conv1_bn")(x)
        x = layers.ReLU(name="conv1_relu")(x)
        x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name='pool_pad')(x)
        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool")(x)
        # x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same", name="pool")(x)

        # stage 1
        stage_name = "stage1"
        filters = 64
        x = ResBlock(filters, strides=(1,1), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)

        # stage 2
        stage_name = "stage2"
        filters = 128
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)

        # stage 3
        stage_name = "stage3"
        filters = 256
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)

        # stage 4
        stage_name = "stage4"
        filters = 512
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x_out = layers.Dense(self.classes_num, activation='softmax', name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="ResNet18", **kwargs)


class ResNet50(KerasModel):
    """ ResNet50 """

    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self, **kwargs):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # input stem
        x = layers.ZeroPadding2D(padding=(3,3), name='conv1_pad')(x_in)
        x = layers.Conv2D(64, (7,7), strides=2, name='conv1_conv')(x)
        x = layers.BatchNormalization(name="conv1_bn")(x)
        x = layers.ReLU(name="conv1_relu")(x)
        x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name='pool_pad')(x)
        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool")(x)
        # x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same", name="pool")(x)

        # stage 1
        stage_name = "stage1"
        filters = 64
        x = ResBlock(filters, strides=(1,1), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)
        x = ResBlock(filters, name=stage_name+"_3")(x)

        # stage 2
        stage_name = "stage2"
        filters = 128
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)
        x = ResBlock(filters, name=stage_name+"_3")(x)
        x = ResBlock(filters, name=stage_name+"_4")(x)

        # stage 3
        stage_name = "stage3"
        filters = 256
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)
        x = ResBlock(filters, name=stage_name+"_3")(x)
        x = ResBlock(filters, name=stage_name+"_4")(x)
        x = ResBlock(filters, name=stage_name+"_5")(x)
        x = ResBlock(filters, name=stage_name+"_6")(x)

        # stage 4
        stage_name = "stage4"
        filters = 512
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)
        x = ResBlock(filters, name=stage_name+"_3")(x)

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x_out = layers.Dense(self.classes_num, activation='softmax', name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="ResNet50", **kwargs)


class ResNet101(KerasModel):
    """ ResNet101 """

    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self, **kwargs):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # input stem
        x = layers.ZeroPadding2D(padding=(3,3), name='conv1_pad')(x_in)
        x = layers.Conv2D(64, (7,7), strides=2, name='conv1_conv')(x)
        x = layers.BatchNormalization(name="conv1_bn")(x)
        x = layers.ReLU(name="conv1_relu")(x)
        x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name='pool_pad')(x)
        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool")(x)
        # x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same", name="pool")(x)

        # stage 1
        stage_name = "stage1"
        filters = 64
        x = ResBlock(filters, strides=(1,1), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)
        x = ResBlock(filters, name=stage_name+"_3")(x)

        # stage 2
        stage_name = "stage2"
        filters = 128
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)
        x = ResBlock(filters, name=stage_name+"_3")(x)
        x = ResBlock(filters, name=stage_name+"_4")(x)

        # stage 3
        stage_name = "stage3"
        filters = 256
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)
        x = ResBlock(filters, name=stage_name+"_3")(x)
        x = ResBlock(filters, name=stage_name+"_4")(x)
        x = ResBlock(filters, name=stage_name+"_5")(x)
        x = ResBlock(filters, name=stage_name+"_6")(x)
        x = ResBlock(filters, name=stage_name+"_7")(x)
        x = ResBlock(filters, name=stage_name+"_8")(x)
        x = ResBlock(filters, name=stage_name+"_9")(x)
        x = ResBlock(filters, name=stage_name+"_10")(x)
        x = ResBlock(filters, name=stage_name+"_11")(x)
        x = ResBlock(filters, name=stage_name+"_12")(x)
        x = ResBlock(filters, name=stage_name+"_13")(x)
        x = ResBlock(filters, name=stage_name+"_14")(x)
        x = ResBlock(filters, name=stage_name+"_15")(x)
        x = ResBlock(filters, name=stage_name+"_16")(x)
        x = ResBlock(filters, name=stage_name+"_17")(x)
        x = ResBlock(filters, name=stage_name+"_18")(x)
        x = ResBlock(filters, name=stage_name+"_19")(x)
        x = ResBlock(filters, name=stage_name+"_20")(x)
        x = ResBlock(filters, name=stage_name+"_21")(x)
        x = ResBlock(filters, name=stage_name+"_22")(x)
        x = ResBlock(filters, name=stage_name+"_23")(x)

        # stage 4
        stage_name = "stage4"
        filters = 512
        x = ResBlock(filters, strides=(2,2), conv_shortcut=True, name=stage_name+"_1")(x)
        x = ResBlock(filters, name=stage_name+"_2")(x)
        x = ResBlock(filters, name=stage_name+"_3")(x)

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x_out = layers.Dense(self.classes_num, activation='softmax', name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="ResNet101", **kwargs)
