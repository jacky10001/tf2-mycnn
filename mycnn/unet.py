# -*- coding: utf-8 -*-
""" U-Net Module

- 完全依照論文架構實現 (unpadded convolutions)
- 建立自定義層 CroppedFeatureMap，來裁切特徵圖

Note:
目前尚未進行訓練，不知道可不可成功訓練網路
如果不可用，會再改成 padded convolutions
確保跳接(skip-connection)的特徵圖大小相同

Refer:
- [U-Net: Convolutional Networks for Biomedical Image Segmentation]
  (https://arxiv.org/abs/1505.04597) (MICCAI 2015)
"""

import tensorflow as tf
from tensorflow.keras import layers
from .core.base_model import KerasModel


class CroppedFeatureMap(layers.Layer):
    """ 裁剪特徵圖中心層

    將特徵圖的中心部分裁剪為目標大小，如果目標大於特徵圖大小，將會報錯
    
    refer:
    https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/layers/Layer
    https://github.com/keras-team/keras/blob/v2.7.0/keras/layers/preprocessing/image_preprocessing.py#L127-L189
    """
    def __init__(self, height, width, **kwargs):
        self.height = height
        self.width = width
        super(CroppedFeatureMap, self).__init__(**kwargs)
    
    def call(self, inputs):
        input_shape = tf.keras.backend.int_shape(inputs)
        if (input_shape[1] < self.height):
            raise Exception(f"target size error: {input_shape[2]} < {self.width}, "
                            "Feature maps size must be bigger than target size, "
                            "please check argument of `height`.")
        
        if (input_shape[2] < self.width):
            raise Exception(f"target size error: {input_shape[2]} < {self.width}. "
                            "Feature maps size must be bigger than target size, "
                            "please check argument of `width`.")
        
        input_shape = tf.shape(inputs)
        h_diff = input_shape[1] - self.height
        w_diff = input_shape[2] - self.width
        
        h_start = tf.cast(h_diff / 2, tf.int32)
        w_start = tf.cast(w_diff / 2, tf.int32)
        outputs = tf.image.crop_to_bounding_box(inputs, h_start, w_start,
                                                 self.height, self.width)
        return tf.cast(outputs, inputs.dtype)
        
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[1] = self.height
        input_shape[2] = self.width
        return tf.TensorShape(input_shape)
        
    def get_config(self):
        config = {
            'height': self.height,
            'width': self.width,
        }
        base_config = super(CroppedFeatureMap, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UNet(KerasModel):
    """ UNet
    UNet 與 FCN 極為相似，但是 UNet 的架構更是強化上採樣的資訊
    核心思想在於使用使用幾乎對稱的跳接方式連接特徵圖，去加強上採樣的資訊

    論文貢獻在於使用編碼解碼器的概念，進行圖像重建(還原)
    跳接能夠使圖像在解碼過程中，得以還原出邊緣高頻資訊
    同時，也能夠避免因為層數過多導致梯度消失 (因為論文等於做了4次pooling)

    Note:
    網路上的 UNet 都會使用 padded convolutions
    這裡我照著論文架構去實現，因此可以完全對上論文的特徵圖大小

    單純用 padded convolutions 其實實作上更簡單，特徵圖大小計算更方便
    """
    def __init__(self,
                 input_shape=(572, 572, 1),
                 classes_num=2,
                 **kwargs):
        self.input_shape = input_shape
        self.classes_num = classes_num
        super().__init__(**kwargs)
      
    def build(self):
        x_in = layers.Input(shape=self.input_shape, name="image")

        # Common parameter for layer
        kernel_size = (3, 3)
        down_pool_size = (2, 2)
        up_sample_size = (2, 2)
        kernel_initializer = "he_normal"

        skip_list = []

        # encoder block 1
        block_name = "encoder1"
        filters = 64
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(x_in)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        pool2x2 = layers.MaxPooling2D(pool_size=down_pool_size, name=block_name+"_pool")(conv3x3)
        skip_list.append(CroppedFeatureMap(392,392)(conv3x3))

        # encoder block 2
        block_name = "encoder2"
        filters = 128
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(pool2x2)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        pool2x2 = layers.MaxPooling2D(pool_size=down_pool_size, name=block_name+"_pool")(conv3x3)
        skip_list.append(CroppedFeatureMap(200,200)(conv3x3))

        # encoder block 3
        block_name = "encoder3"
        filters = 256
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(pool2x2)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        pool2x2 = layers.MaxPooling2D(pool_size=down_pool_size, name=block_name+"_pool")(conv3x3)
        skip_list.append(CroppedFeatureMap(104,104)(conv3x3))

        # encoder block 4
        block_name = "encoder4"
        filters = 512
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(pool2x2)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        pool2x2 = layers.MaxPooling2D(pool_size=down_pool_size, name=block_name+"_pool")(conv3x3)
        skip_list.append(CroppedFeatureMap(56,56)(conv3x3))

        # encoder block 5
        block_name = "bottom"
        filters = 1024
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(pool2x2)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        
        block_name = "decoder4"
        filters = 512
        conv3x3 = layers.Conv2DTranspose(filters, up_sample_size, strides=up_sample_size, name=block_name+"_conv1t")(conv3x3)
        conv3x3 = layers.Concatenate(name=block_name+"_skip")([conv3x3, skip_list[-1]])
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        
        block_name = "decoder3"
        filters = 256
        conv3x3 = layers.Conv2DTranspose(filters, up_sample_size, strides=up_sample_size, name=block_name+"_conv1t")(conv3x3)
        conv3x3 = layers.Concatenate(name=block_name+"_skip")([conv3x3, skip_list[-2]])
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        
        block_name = "decoder2"
        filters = 128
        conv3x3 = layers.Conv2DTranspose(filters, up_sample_size, strides=up_sample_size, name=block_name+"_conv1t")(conv3x3)
        conv3x3 = layers.Concatenate(name=block_name+"_skip")([conv3x3, skip_list[-3]])
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        
        block_name = "decoder1"
        filters = 64
        conv3x3 = layers.Conv2DTranspose(filters, up_sample_size, strides=up_sample_size, name=block_name+"_conv1t")(conv3x3)
        conv3x3 = layers.Concatenate(name=block_name+"_skip")([conv3x3, skip_list[-4]])
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv1")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn1")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu1")(conv3x3)
        conv3x3 = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, name=block_name+"_conv2")(conv3x3)
        conv3x3 = layers.BatchNormalization(name=block_name+"_bn2")(conv3x3)
        conv3x3 = layers.ReLU(name=block_name+"_relu2")(conv3x3)
        conv1x1 = layers.Conv2D(self.classes_num, (1,1), kernel_initializer=kernel_initializer, name=block_name+"_conv3")(conv3x3)

        x_out = layers.Softmax(name="prediction")(conv1x1)
        
        self.setup_model(x_in, x_out, name="UNet")