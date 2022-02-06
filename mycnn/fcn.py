# -*- coding: utf-8 -*-
""" FCN Module

分成兩種版本
- 使用 Keras Layers API 自行建構 VGG16 骨幹網路
- 使用 Keras Applications VGG16 API
  建構 VGG16 骨幹網路並載入 Pre-training Weights

Refer:
- [Fully Convolutional Networks for Semantic Segmentation]
  (https://arxiv.org/abs/1411.4038) (CVPR 2015, IEEE/CVF)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition]
  (https://arxiv.org/abs/1409.1556) (ICLR 2015)
- [kevinddchen/Keras-FCN]
  (https://github.com/kevinddchen/Keras-FCN)
"""

import tensorflow as tf
from tensorflow.keras import layers
from .core.base_model import KerasModel


def my_vgg16(x_in, filters_list: list=None, use_bn: bool=False) -> list:
    """ my_vgg16
    使用 Keras Layers API 自行建構 VGG16 骨幹網路
    全連接層 (fully connection) 使用卷積層來實現

    Arguments
    x_in: 輸入 Tensor (tf.keras.layers.Input)
    filters_list: 設置每個 block 層的 filters 數量

    Return
    pool_tensor: 回傳 pool list，包含 block3_pool、block4_pool、最終輸出
    """
    if isinstance(filters_list, list):
        if len(filters_list) != 6:
            raise ValueError("Error: filters_list length must be 6.")

    pool_tensor = []

    # Common parameter for layer
    kernel_size = (3, 3)
    pool_size = (2, 2)
    padding = "same"
    kernel_initializer = "he_normal"

    # vgg block 1
    block_name = "block1"
    filters = 64 if not filters_list else filters_list[0]
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x_in)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

    # vgg block 2
    block_name = "block2"
    filters = 128 if not filters_list else filters_list[1]
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

    # vgg block 3
    block_name = "block3"
    filters = 256 if not filters_list else filters_list[2]
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv3")(x)
    x = layers.BatchNormalization(name=block_name+"_bn3")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu3")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)
    pool_tensor.append(x)

    # vgg block 4
    block_name = "block4"
    filters = 512 if not filters_list else filters_list[3]
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv3")(x)
    x = layers.BatchNormalization(name=block_name+"_bn3")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu3")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)
    pool_tensor.append(x)

    # vgg block 5
    block_name = "block5"
    filters = 512 if not filters_list else filters_list[4]
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu2")(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv3")(x)
    x = layers.BatchNormalization(name=block_name+"_bn3")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu3")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name=block_name+"_pool")(x)

    # fully connection based on convolution
    block_name = "fc"
    filters = 4096 if not filters_list else filters_list[5]
    x = layers.Conv2D(filters, (7,7), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
    x = layers.BatchNormalization(name=block_name+"_bn1")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu1")(x)
    x = layers.Dropout(0.5, name=block_name+"_dropout")(x)
        
    x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
    x = layers.BatchNormalization(name=block_name+"_bn2")(x) if use_bn else x
    x = layers.ReLU(name=block_name+"_relu2")(x)
    pool_tensor.append(x)

    return pool_tensor


class FCN32(KerasModel):
    """ FCN32 """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=256,
                 filters_list=None,
                 use_bn=False,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        self.filters_list = filters_list
        self.use_bn = use_bn
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        block_name = "fcn32"
        padding = "same"
        kernel_initializer = "he_normal"

        x_in = layers.Input(shape=self.input_shape, name="image")

        _, _, x = my_vgg16(x_in, filters_list=self.filters_list, use_bn=self.use_bn)
        
        x = layers.Conv2D(self.classes_num, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(self.classes_num, (32, 32), strides=(32, 32), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = tf.image.resize(x, self.input_shape[:2], method="bilinear")
        x = layers.Reshape((-1, self.classes_num))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN32")


class FCN16(KerasModel):
    """ FCN16 """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=256,
                 filters_list=None,
                 use_bn=False,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        self.filters_list = filters_list
        self.use_bn = use_bn
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        block_name = "fcn16"
        padding = "same"
        kernel_initializer = "he_normal"
        
        x_in = layers.Input(shape=self.input_shape, name="image")

        pool3, pool4, x = my_vgg16(x_in, filters_list=self.filters_list, use_bn=self.use_bn)

        pool4 = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool4')(pool4)
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(self.classes_num, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Add(name=block_name+"_up_x2")([x, pool4])

        pool3 = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool3')(pool3)
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv2')(x)
        x = layers.Conv2DTranspose(self.classes_num, (16,16), strides=(16,16), padding="valid", use_bias=False, name=block_name+"_conv2t")(x)
        x = tf.image.resize(x, self.input_shape[:2], method="bilinear")
        x = layers.Reshape((-1, self.classes_num))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN16")


class FCN8(KerasModel):
    """ FCN8 """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=256,
                 filters_list=None,
                 use_bn=False,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        self.filters_list = filters_list
        self.use_bn = use_bn
        super().__init__(**kwargs)
      
    def build(self):
        # Common parameter for layer
        block_name = "fcn8"
        padding = "same"
        kernel_initializer = "he_normal"

        x_in = layers.Input(shape=self.input_shape, name="image")

        pool3, pool4, x = my_vgg16(x_in, filters_list=self.filters_list, use_bn=self.use_bn)

        pool4 = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool4')(pool4)
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(self.classes_num, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Add(name=block_name+"_up_x2")([x, pool4])

        pool3 = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool3')(pool3)
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv2')(x)
        x = layers.Conv2DTranspose(self.classes_num, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv2t")(x)
        x = layers.Add(name=block_name+"_up_x4")([x, pool3])
        
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv3')(x)
        x = layers.Conv2DTranspose(self.classes_num, (8,8), strides=(8,8), padding="valid", use_bias=False, name=block_name+"_conv3t")(x)
        x = tf.image.resize(x, self.input_shape[:2], method="bilinear")
        x = layers.Reshape((-1, self.classes_num))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN8")


class FCN32_KERAS(KerasModel):
    """ FCN32_KERAS
    基於 tf.keras.applications.VGG16 作為骨幹網路
    """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=256,
                 top_filters=4096,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        self.top_filters = top_filters
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
        filters = self.top_filters
        x = layers.Conv2D(filters, (7,7), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)
        
        x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        
        # fcn
        block_name = "fcn32"
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(self.classes_num, (32, 32), strides=(32, 32), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = tf.image.resize(x, self.input_shape[:2], method="bilinear")
        x = layers.Reshape((-1, self.classes_num))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN32")


class FCN16_KERAS(KerasModel):
    """ FCN16_KERAS
    基於 tf.keras.applications.VGG16 作為骨幹網路
    """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=256,
                 top_filters=4096,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        self.top_filters = top_filters
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
        filters = self.top_filters
        x = layers.Conv2D(filters, (7,7), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(pool5)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)
        
        x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        
        # fcn
        block_name = "fcn16"
        pool4 = backbone.get_layer("block4_pool").output
        pool4 = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool4')(pool4)
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(self.classes_num, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Add(name=block_name+"_up_x2")([x, pool4])

        pool3 = backbone.get_layer("block3_pool").output
        pool3 = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool3')(pool3)
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv2')(x)
        x = layers.Conv2DTranspose(self.classes_num, (16,16), strides=(16,16), padding="valid", use_bias=False, name=block_name+"_conv2t")(x)
        x = tf.image.resize(x, self.input_shape[:2], method="bilinear")
        x = layers.Reshape((-1, self.classes_num))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN16")


class FCN8_KERAS(KerasModel):
    """ FCN8_KERAS
    基於 tf.keras.applications.VGG16 作為骨幹網路
    """
    def __init__(self,
                 input_shape=(224, 224, 3),
                 classes_num=256,
                 top_filters=4096,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        self.top_filters = top_filters
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
        filters = self.top_filters
        x = layers.Conv2D(filters, (7,7), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(pool5)
        x = layers.ReLU(name=block_name+"_relu1")(x)
        x = layers.Dropout(0.5, name=block_name+"_dropout")(x)
        
        x = layers.Conv2D(filters, (1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(x)
        x = layers.ReLU(name=block_name+"_relu2")(x)
        
        # fcn
        block_name = "fcn8"
        pool4 = backbone.get_layer("block4_pool").output
        pool4 = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool4')(pool4)
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv1')(x)
        x = layers.Conv2DTranspose(self.classes_num, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv1t")(x)
        x = layers.Add(name=block_name+"_up_x2")([x, pool4])

        pool3 = backbone.get_layer("block3_pool").output
        pool3 = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_pool3')(pool3)
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv2')(x)
        x = layers.Conv2DTranspose(self.classes_num, (2,2), strides=(2,2), padding="valid", use_bias=False, name=block_name+"_conv2t")(x)
        x = layers.Add(name=block_name+"_up_x4")([x, pool3])
        
        x = layers.Conv2D(self.classes_num, kernel_size=(1,1), padding=padding, kernel_initializer=kernel_initializer, name=block_name+'_conv3')(x)
        x = layers.Conv2DTranspose(self.classes_num, (8,8), strides=(8,8), padding="valid", use_bias=False, name=block_name+"_conv3t")(x)
        x = tf.image.resize(x, self.input_shape[:2], method="bilinear")
        x = layers.Reshape((-1, self.classes_num))(x)
        x_out = layers.Softmax(name="predictions")(x)
        
        self.setup_model(x_in, x_out, name="FCN8")