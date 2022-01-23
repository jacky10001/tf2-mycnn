# -*- coding: utf-8 -*-
""" FCN Module

分成兩種版本
- 使用 Keras Layers API 自行建構 VGG16 骨幹網路
- 使用 Keras Applications API，並載入 Pre-training Weights

Refer:
- [Fully Convolutional Networks for Semantic Segmentation]
  (https://arxiv.org/abs/1411.4038) (CVPR 2015, IEEE/CVF)
"""

import tensorflow as tf
from tensorflow.keras import layers
from .core.base_model import KerasModel


def my_vgg16(x_in) -> list:
    """ my_vgg16
    使用 Keras Layers API 自行建構 VGG16 骨幹網路
    全連接層 (fully connection) 使用卷積層來實現

    Arguments
    x_in: 輸入 Tensor (tf.keras.layers.Input)

    Return
    pool_tensor: 回傳 pool list，包含 block3_pool、block4_pool、最終輸出
    """

    pool_tensor = []

    # Common parameter for layer
    kernel_size = (3, 3)
    pool_size = (2, 2)
    padding = "same"
    kernel_initializer = "he_normal"

    # vgg block 1
    block_name = "block1"
    filters = 64
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x_in)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x)
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x)
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

    # vgg block 2
    block_name = "block2"
    filters = 128
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x)
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x)
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

    # vgg block 3
    block_name = "block3"
    filters = 256
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x)
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x)
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv3")(x)
    x = layers.BatchNormalization(name=block_name+"_bn3")(x)
    x = layers.ReLU(name=block_name+"_relu3")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)
    pool_tensor.append(x)

    # vgg block 4
    block_name = "block4"
    filters = 512
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x)
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x)
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv3")(x)
    x = layers.BatchNormalization(name=block_name+"_bn3")(x)
    x = layers.ReLU(name=block_name+"_relu3")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)
    pool_tensor.append(x)

    # vgg block 5
    block_name = "block5"
    filters = 512
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x)
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x)
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv3")(x)
    x = layers.BatchNormalization(name=block_name+"_bn3")(x)
    x = layers.ReLU(name=block_name+"_relu3")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

    # fully connection based on convolution
    block_name = "fc"
    filters = 4096
    x = layers.Conv2D(filters, (7,7), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x)
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Dropout(0.5, name=block_name+"_dropout")(x)
        
    x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x)
    x = layers.ReLU(name=block_name+"_relu2")(x)
    pool_tensor.append(x)

    return pool_tensor


class FCN32(KerasModel):
    """ FCN32 """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        block_name = "fcn32"
        filters = self.classes_num
        padding = "same"
        kernel_initializer = "he_normal"

        x_in = layers.Input(shape=self.input_shape, name="image")

        _, _, x = my_vgg16(x_in)
        
        x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(filters, (32, 32), strides=(32, 32), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Reshape((-1, filters))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN32")


class FCN16(KerasModel):
    """ FCN16 """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        block_name = "fcn16"
        filters = self.classes_num
        padding = "same"
        kernel_initializer = "he_normal"
        
        x_in = layers.Input(shape=self.input_shape, name="image")

        pool3, pool4, x = my_vgg16(x_in)

        pool4 = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool4')(pool4)
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Add(name=block_name+"_up_x2")([x, pool4])

        pool3 = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool3')(pool3)
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv2')(x)
        x = layers.Conv2DTranspose(filters, (16,16), strides=(16,16), padding="valid", use_bias=False, name=block_name+"_conv2t")(x)
        x = layers.Reshape((-1, filters))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN16")


class FCN8(KerasModel):
    """ FCN8 """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        block_name = "fcn8"
        filters = self.classes_num
        padding = "same"
        kernel_initializer = "he_normal"

        x_in = layers.Input(shape=self.input_shape, name="image")

        pool3, pool4, x = my_vgg16(x_in)

        pool4 = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool4')(pool4)
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Add(name=block_name+"_up_x2")([x, pool4])

        pool3 = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool3')(pool3)
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv2')(x)
        x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv2t")(x)
        x = layers.Add(name=block_name+"_up_x4")([x, pool3])
        
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv3')(x)
        x = layers.Conv2DTranspose(filters, (8,8), strides=(8,8), padding="valid", use_bias=False, name=block_name+"_conv3t")(x)
        x = layers.Reshape((-1, filters))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN8")


class FCN32_KERAS(KerasModel):
    """ FCN32_KERAS
    基於 tf.keras.applications.VGG16 作為骨幹網路
    """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        kernel_initializer = "he_normal"
        padding = "same"

        x_in = layers.Input(shape=self.input_shape, name="image")
        backbone = tf.keras.applications.VGG16(include_top=False, input_tensor=x_in)
        x = backbone.output

        # fc
        block_name = "fc"
        filters = 4096
        x = layers.Conv2D(filters, (7,7), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)
        
        x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        
        # fcn
        block_name = "fcn32"
        filters = self.classes_num
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(filters, (32, 32), strides=(32, 32), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Reshape((-1, filters))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN32")


class FCN16_KERAS(KerasModel):
    """ FCN16_KERAS
    基於 tf.keras.applications.VGG16 作為骨幹網路
    """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        kernel_initializer = "he_normal"
        padding = "same"

        x_in = layers.Input(shape=self.input_shape, name="image")
        backbone = tf.keras.applications.VGG16(include_top=False, input_tensor=x_in)
        pool5 = backbone.output

        # fc
        block_name = "fc"
        filters = 4096
        x = layers.Conv2D(filters, (7,7), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(pool5)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)
        
        x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        
        # fcn
        block_name = "fcn16"
        filters = self.classes_num

        pool4 = backbone.get_layer("block4_pool").output
        pool4 = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool4')(pool4)
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Add(name=block_name+"_up_x2")([x, pool4])

        pool3 = backbone.get_layer("block3_pool").output
        pool3 = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool3')(pool3)
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv2')(x)
        x = layers.Conv2DTranspose(filters, (16,16), strides=(16,16), padding="valid", use_bias=False, name=block_name+"_conv2t")(x)
        x = layers.Reshape((-1, filters))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN16")


class FCN8_KERAS(KerasModel):
    """ FCN8_KERAS
    基於 tf.keras.applications.VGG16 作為骨幹網路
    """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=1000,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        kernel_initializer = "he_normal"
        padding = "same"

        x_in = layers.Input(shape=self.input_shape, name="image")
        backbone = tf.keras.applications.VGG16(include_top=False, input_tensor=x_in)
        pool5 = backbone.output

        # fc
        block_name = "fc"
        filters = 4096
        x = layers.Conv2D(filters, (7,7), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(pool5)
        x = layers.BatchNormalization(name=block_name+"_bn1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)
        
        x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
        x = layers.BatchNormalization(name=block_name+"_bn2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        
        # fcn
        block_name = "fcn8"
        filters = self.classes_num

        pool4 = backbone.get_layer("block4_pool").output
        pool4 = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool4')(pool4)
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Add(name=block_name+"_up_x2")([x, pool4])

        pool3 = backbone.get_layer("block3_pool").output
        pool3 = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool3')(pool3)
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv2')(x)
        x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv2t")(x)
        x = layers.Add(name=block_name+"_up_x4")([x, pool3])
        
        x = layers.Conv2D(filters, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv3')(x)
        x = layers.Conv2DTranspose(filters, (8,8), strides=(8,8), padding="valid", use_bias=False, name=block_name+"_conv3t")(x)
        x = layers.Reshape((-1, filters))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN8")