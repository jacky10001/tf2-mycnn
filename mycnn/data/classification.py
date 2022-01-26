# -*- coding: utf-8 -*-

"""
基於 tf.data.Dataset API 的資料擴增
這裡我將以前有用 TensorFlow 寫過的影像處理相關函式整理起來
並參考 Keras 官方的預處理函式 image_dataset_from_directory
然後重新實現新的函式 generate_classification_dataset
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


def generate_classification_dataset(directory,
                                    batch_size=32,
                                    image_size=(256, 256),
                                    gray=False,
                                    shuffle=False,
                                    validation_split=None,
                                    **kwargs):
    class_names = []
    file_paths = []
    labels = []

    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            class_names.append(subdir)
    class_indices = dict(zip(class_names, range(len(class_names))))
    print(f"Class indices:\n{class_indices}\n")
    if not class_names:
        raise ValueError("No subdirs found.")

    walk = os.walk(directory)
    for root, _, files in natsorted(walk, key=lambda x: x[0]):
        for fname in natsorted(files):
            if fname.lower().endswith(ALLOWLIST_FORMATS):
                filepath = os.path.join(root, fname)
                file_paths.append(filepath)
                labels.append(class_indices[os.path.basename(root)])
    if not file_paths:
        raise ValueError("No images found.")

    print(f'Found {len(file_paths)} files belonging to {len(class_names)} classes.')

    def load_img(x):
        x = tf.io.read_file(x)
        num_channels = 1 if gray else 3
        x = tf.io.decode_image(x, channels=num_channels, expand_animations=False)
        x = tf.image.resize(x, image_size)
        x = tf.cast(x, tf.float32) / 255.0
        
        if len(x.shape) == 2:
            print("Gray image, add one channel!!!", end=" - ")
            x = tf.expand_dims(x, axis=-1)
        else:
            if x.shape[-1] == 3:
                print("RGB image", end=" - ")
            elif  x.shape[-1] == 1:
                print("Gray image", end=" - ")
        
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
            
        return x

    def load_lbl(y):
        y = tf.one_hot(y, len(class_names))
        return y
    
    if not validation_split:
        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
        img_ds = path_ds.map(load_img)

        lbl_ds = tf.data.Dataset.from_tensor_slices(labels)
        lbl_ds = lbl_ds.map(load_lbl)

        dataset = tf.data.Dataset.zip((img_ds, lbl_ds))
        if shuffle:
            print("Shuffle Data!!!")
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=100)
        dataset = dataset.batch(batch_size)

        dataset.class_names = class_names
        dataset.file_paths = file_paths

        return dataset
    else:
        num_val_samples = int(len(file_paths) * validation_split)
        
        print('Using %d files for training.' % (len(file_paths) - num_val_samples))
        print('Using %d files for validation.' % (num_val_samples))

        tra_path_ds = tf.data.Dataset.from_tensor_slices(file_paths[:num_val_samples])
        val_path_ds = tf.data.Dataset.from_tensor_slices(file_paths[num_val_samples:])
        tra_img_ds = tra_path_ds.map(load_img)
        val_img_ds = val_path_ds.map(load_img)

        tra_lbl_ds = tf.data.Dataset.from_tensor_slices(labels[:num_val_samples])
        val_lbl_ds = tf.data.Dataset.from_tensor_slices(labels[num_val_samples:])
        tra_lbl_ds = tra_lbl_ds.map(load_lbl)
        val_lbl_ds = val_lbl_ds.map(load_lbl)

        tra_dataset = tf.data.Dataset.zip((tra_img_ds, tra_lbl_ds))
        val_dataset = tf.data.Dataset.zip((val_img_ds, val_lbl_ds))
        if shuffle:
            print("Shuffle Data!!!")
            tra_dataset = tra_dataset.shuffle(buffer_size=batch_size * 8, seed=100)
            val_dataset = val_dataset.shuffle(buffer_size=batch_size * 8, seed=100)
        tra_dataset = tra_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        tra_dataset.class_names = class_names
        val_dataset.class_names = class_names
        tra_dataset.file_paths = file_paths[:num_val_samples]
        val_dataset.file_paths = file_paths[num_val_samples:]

        return tra_dataset, val_dataset