# -*- coding: utf-8 -*-

"""
用於讀取分割資料
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


def generate_segmentation_dataset(directory,
                                  batch_size=10,
                                  image_size=(256, 256),
                                  mask_size=None,
                                  classes_num=21,
                                  subtract_mean=None,
                                  divide_stddev=None,
                                  gray=False,
                                  shuffle_filepath=False,
                                  shuffle_dataset=False,
                                  seed=100,
                                  validation_split=None,
                                  mask_mode=0,
                                  **kwargs):
    if mask_mode == 1:
        if not mask_size:
            raise ValueError("Error: Invalid mask size.")
        if image_size[0] < mask_size[0] or image_size[1] < mask_size[1]:
            raise ValueError("Error: Invalid mask size. Image size must be bigger than mask size.")
    else:
        mask_size = image_size if not mask_size else mask_size
    
    image_paths = []
    mask_paths = []

    for subdir in sorted(os.listdir(directory)):
        subdir_fullpath = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_fullpath):
            if not (subdir.lower().startswith("images") or subdir.lower().startswith("masks")):
                raise ValueError("")
    print(f"Read segmentation dataset.\n")

    walk = os.walk(directory)
    for root, _, files in natsorted(walk, key=lambda x: x[0]):
        for fname in natsorted(files):
            if fname.lower().endswith(ALLOWLIST_FORMATS):
                filepath = os.path.join(root, fname)
                if os.path.basename(root).lower().endswith("images"):
                    image_paths.append(filepath)
                if os.path.basename(root).lower().endswith("masks"):
                    mask_paths.append(filepath)
    if not image_paths:
        raise ValueError("No images found.")
    if not mask_paths:
        raise ValueError("No masks found.")
    if len(image_paths) != len(mask_paths):
        raise ValueError("The number of images and masks are not equal.")

    print(f'Found {len(image_paths)} image files and {len(mask_paths)}.')

    if shuffle_filepath:
        rng = np.random.RandomState(seed)
        rng.shuffle(image_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(mask_paths)

    def load_data(x, y, subset: str):
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
                x = tf.cond(
                    tf.random.uniform((), 0, 1) > 0.5,
                    lambda: aug_fn(x), lambda: x)
        else:
            print("Not use data augmentation.")
        
        x = tf.cast(x, tf.float32)
        
        clip_value_min = 0
        clip_value_max = 255

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

        y = tf.io.read_file(y)
        y = tf.io.decode_image(y, channels=1, expand_animations=False)
        y = tf.image.resize(y, mask_size, method="nearest")  # 使用近鄰插植，處理物體邊緣值維持一致
        y = tf.cast(y, tf.uint8)
        y = tf.reshape(y, (-1,))
        y = tf.one_hot(y, classes_num)
        # y = tf.reshape(y, mask_size+(classes_num,))
        return x, y

    def load_ori_unet_mask(y):
        y = tf.io.read_file(y)
        y = tf.io.decode_image(y, channels=1, expand_animations=False)
        y = tf.image.resize(y, image_size, method="nearest")  # 使用近鄰插植，處理物體邊緣值維持一致
        h_diff = image_size[0] - mask_size[0]
        w_diff = image_size[1] - mask_size[0]
        h_start = tf.cast(h_diff/2, tf.int32)
        w_start = tf.cast(w_diff/2, tf.int32)
        y = tf.image.crop_to_bounding_box(y, h_start, w_start, mask_size[0], mask_size[1])
        y = tf.cast(y, tf.uint8)
        y = tf.reshape(y, (-1,))
        y = tf.one_hot(y, classes_num)
        # y = tf.reshape(y, mask_size+(classes_num,))
        return y
    
    if not validation_split:
        img_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        lbl_oath_ds = tf.data.Dataset.from_tensor_slices(mask_paths)
        dataset = tf.data.Dataset.zip((img_path_ds, lbl_oath_ds))

        if shuffle_dataset:
            print("Shuffle dataset!!!")
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=100)
        
        autotune = tf.data.AUTOTUNE
        dataset = dataset.map(lambda x, y: load_data(x, y, "all"), num_parallel_calls=autotune)

        dataset = dataset.batch(batch_size)

        dataset.image_paths = image_paths
        dataset.mask_paths = mask_paths
        dataset.batch_size = batch_size

        return dataset
    else:
        num_tra_samples = int(len(image_paths)*(1-validation_split))
        
        print("Using %d files for training."%(num_tra_samples))
        print("Using %d files for validation."%(len(image_paths)-num_tra_samples))

        tra_img_path_ds = tf.data.Dataset.from_tensor_slices(image_paths[:num_tra_samples])
        tra_lbl_path_ds = tf.data.Dataset.from_tensor_slices(mask_paths[:num_tra_samples])
        tra_dataset = tf.data.Dataset.zip((tra_img_path_ds, tra_lbl_path_ds))

        val_img_path_ds = tf.data.Dataset.from_tensor_slices(image_paths[num_tra_samples:])
        val_lbl_path_ds = tf.data.Dataset.from_tensor_slices(mask_paths[num_tra_samples:])
        val_dataset = tf.data.Dataset.zip((val_img_path_ds, val_lbl_path_ds))

        if shuffle_dataset:
            print("Shuffle training dataset!!!")
            tra_dataset = tra_dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        
        autotune = tf.data.AUTOTUNE
        tra_dataset = tra_dataset.map(lambda x, y: load_data(x, y, "train"), num_parallel_calls=autotune)
        val_dataset = val_dataset.map(lambda x, y: load_data(x, y, "valid"), num_parallel_calls=autotune)

        tra_dataset = tra_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        tra_dataset.image_paths = image_paths[:num_tra_samples]
        val_dataset.image_paths = image_paths[num_tra_samples:]
        tra_dataset.mask_paths = mask_paths[:num_tra_samples]
        val_dataset.mask_paths = mask_paths[num_tra_samples:]
        tra_dataset.batch_size = batch_size
        val_dataset.batch_size = batch_size

        return tra_dataset, val_dataset