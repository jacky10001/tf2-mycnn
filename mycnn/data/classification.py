# -*- coding: utf-8 -*-

"""
基於 tf.data.Dataset API 的資料擴增
這裡我將以前有用 TensorFlow 寫過的影像處理相關函式整理起來
並參考 Keras 官方的預處理函式 image_dataset_from_directory
然後重新實現新的函式 generate_classification_dataset
"""

import os
import glob
import numpy as np
from natsort import natsorted
import tensorflow as tf

ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def flip_h(x, **kwargs):
    seed = tf.random.uniform([2], 1, 10000, dtype=tf.int32)
    x = tf.image.stateless_random_flip_left_right(x, seed=seed)
    return x


def flip_v(x, **kwargs):
    seed = tf.random.uniform([2], 1, 10000, dtype=tf.int32)
    x = tf.image.stateless_random_flip_up_down(x, seed=seed)
    return x


def rotate(x, **kwargs):
    k = tf.random.uniform([], 1, 4, tf.int32)
    x = tf.image.rot90(x, k)
    return x


def hue(x, max_delta=0.08, **kwargs):  # 色調
    seed = tf.random.uniform([2], 1, 10000, dtype=tf.int32)
    x = tf.image.stateless_random_hue(x, max_delta, seed=seed)
    return x


def brightness(x, max_delta=0.5, **kwargs):  # 亮度
    seed = tf.random.uniform([2], 1, 10000, dtype=tf.int32)
    x = tf.image.stateless_random_brightness(x, max_delta, seed=seed)
    return x


def saturation(x, lower=0.6, upper=1.6, **kwargs):  # 飽和度
    seed = tf.random.uniform([2], 1, 10000, dtype=tf.int32)
    x = tf.image.stateless_random_saturation(x, lower, upper, seed=seed)
    return x


def contrast(x, lower=0.7, upper=1.3, **kwargs):  # 對比度
    seed = tf.random.uniform([2], 1, 10000, dtype=tf.int32)
    x = tf.image.stateless_random_contrast(x, lower, upper, seed=seed)
    return x


def zoom_scale(x, scale_minval=0.5, scale_maxval=1.5, **kwargs):
    height, width, _ = x.shape
    scale = tf.random.uniform([], minval=scale_minval, maxval=scale_maxval)
    new_size = (scale*height, scale*width)
    x = tf.image.resize(x, new_size)
    x = tf.image.resize_with_crop_or_pad(x, height, width)
    x = tf.cast(x, tf.uint8)
    return x


def generate_classification_dataset(directory,
                                    batch_size=32,
                                    image_size=(256, 256),
                                    subtract_mean=None,
                                    divide_stddev=None,
                                    gray=False,
                                    shuffle_filepath=False,
                                    shuffle_dataset=False,
                                    seed=100,
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

    if shuffle_filepath:
        rng = np.random.RandomState(seed)
        rng.shuffle(file_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(labels)

    def load_img(x, subset: str):
        num_channels = 1 if gray else 3
        x = tf.io.read_file(x)
        x = tf.io.decode_image(x, channels=num_channels, expand_animations=False)
        x = tf.image.resize(x, image_size)
        x = tf.cast(x, tf.uint8)
        
        print(f"\n{subset} - ", end="")
        if len(x.shape) == 2:
            print("Gray image, add one channel. - ", end="")
            x = tf.expand_dims(x, axis=-1)
        else:
            if x.shape[-1] == 3:
                print("RGB image - ", end="")
            elif  x.shape[-1] == 1:
                print("Gray image - ", end="")
        
        if kwargs and subset.startswith("train"):
            print("Use data augmentation.")
            
            for k in kwargs.keys():
                print(" "*5, k, ":", kwargs[k])
                aug_fn = eval(k)
                x = aug_fn(x)
        else:
            print("Not use data augmentation.")
        
        x = tf.cast(x, tf.float32)
        
        clip_value_min = 0.
        clip_value_max = 255.

        if subtract_mean is not None:
            x -= subtract_mean
            clip_value_min -= subtract_mean
            clip_value_max -= subtract_mean

        if divide_stddev is not None:
            x /= divide_stddev
            clip_value_min /= divide_stddev
            clip_value_max /= divide_stddev
        
        print(f"Rescale value to [{clip_value_min}, {clip_value_max}].")
        x = tf.clip_by_value(x, clip_value_min, clip_value_max)
            
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
        if shuffle_dataset:
            print("Shuffle dataset!!!")
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=100)
        dataset = dataset.batch(batch_size)

        dataset.class_names = class_names
        dataset.file_paths = file_paths

        return dataset
    else:
        num_tra_samples = int(len(file_paths)*(1-validation_split))
        
        print("Using %d files for training."%(num_tra_samples))
        print("Using %d files for validation."%(len(file_paths)-num_tra_samples))

        tra_path_ds = tf.data.Dataset.from_tensor_slices(file_paths[:num_tra_samples])
        val_path_ds = tf.data.Dataset.from_tensor_slices(file_paths[num_tra_samples:])
        tra_img_ds = tra_path_ds.map(lambda x: load_img(x, "train"))
        val_img_ds = val_path_ds.map(lambda x: load_img(x, "valid"))

        tra_lbl_ds = tf.data.Dataset.from_tensor_slices(labels[:num_tra_samples])
        val_lbl_ds = tf.data.Dataset.from_tensor_slices(labels[num_tra_samples:])
        tra_lbl_ds = tra_lbl_ds.map(load_lbl)
        val_lbl_ds = val_lbl_ds.map(load_lbl)

        tra_dataset = tf.data.Dataset.zip((tra_img_ds, tra_lbl_ds))
        val_dataset = tf.data.Dataset.zip((val_img_ds, val_lbl_ds))
        if shuffle_dataset:
            print("Shuffle training dataset!!!")
            tra_dataset = tra_dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        tra_dataset = tra_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        tra_dataset.class_names = class_names
        val_dataset.class_names = class_names
        tra_dataset.file_paths = file_paths[:num_tra_samples]
        val_dataset.file_paths = file_paths[num_tra_samples:]

        return tra_dataset, val_dataset