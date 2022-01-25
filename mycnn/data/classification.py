# -*- coding: utf-8 -*-

"""
基於 TensorFlow API 的資料擴增
需搭配 tf.data.Dataset map 函式
"""

import os
import glob
from natsort import natsorted
import tensorflow as tf


ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')


def flip_h(x):
    x = tf.image.random_flip_left_right(x)
    return x


def flip_v(x):
    x = tf.image.random_flip_up_down(x)
    return x


def rotate(x):
    k = tf.random.uniform([], 1, 4, tf.int32)
    x = tf.image.rot90(x, k)
    return x


def hue(x, val=0.08):  # 色調
    x = tf.image.random_hue(x, val)
    return x


def brightness(x, val=0.05):  # 亮度
    x = tf.image.random_brightness(x, val)
    return x


def saturation(x, minval=0.6, maxval=1.6):  # 飽和度
    x = tf.image.random_saturation(x, minval, maxval)
    return x


def contrast(x, minval=0.7, maxval=1.3):  # 對比度
    x = tf.image.random_contrast(x, minval, maxval)
    return x


def color(x):
    x = hue(x)
    x = saturation(x)
    x = brightness(x)
    x = contrast(x)
    return x


def zoom_scale(x, scale_minval=0.5, scale_maxval=1.5):
    height, width, channel = x.shape
    scale = tf.random.uniform([], scale_minval, scale_maxval)
    new_size = (scale*height, scale*width)
    x = tf.image.resize(x, new_size)
    x = tf.image.resize_with_crop_or_pad(x, height, width)
    return x


def parse_fn(directory, gray=False, **kwargs):
    filepaths = []
    labels = []
    
    # inferred_class_names = []
    # for subdir in sorted(os.listdir(directory)):
    #     if os.path.isdir(os.path.join(directory, subdir)):
    #         inferred_class_names.append(subdir)
    # class_indices = dict(zip(class_names, range(len(class_names))))

    walk = os.walk(directory)
    for root, _, files in natsorted(walk, key=lambda x: x[0]):
        for fname in natsorted(files):
            if fname.lower().endswith(ALLOWLIST_FORMATS):
                filepath = os.path.join(root, fname)
                filepaths.append(filepath)
    
    def wraper(x, y):
        if gray:
            x = tf.image.rgb_to_grayscale(x)
        x = tf.cast(x, tf.float32) / 255.0
        
        if len(x.shape) == 2:
            print("\nGray image, add one channel!!!")
            x = tf.expand_dims(x, axis=-1)
        else:
            if x.shape[-1] == 3:
                print("\nRGB image")
            elif  x.shape[-1] == 1:
                print("\nGray image")
        print(x.shape)
        
        if kwargs:
            print("Data Augmentation!!!")
            
            for k in kwargs.keys():
                print(" "*5, k)
                aug_fn = eval(k)
                x = tf.cond(
                    tf.random.uniform((), 0, 1) > 0.5,
                    lambda: aug_fn(x), lambda: x)
        else:
            print("Not Data Augmentation!!!")
            
        y = tf.one_hot(y, 10)
        return x, y
    return wraper