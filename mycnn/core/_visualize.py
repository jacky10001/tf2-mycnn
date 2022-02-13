# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


def _cal_images_per_row(layer_channel):
    """計算特徵圖要輸出的列數"""
    images_per_row = 1
    if layer_channel > 4:
        if layer_channel%2 == 0:
            images_per_row = np.gcd(layer_channel, 16)
        elif layer_channel%3 == 0:
            images_per_row = np.gcd(layer_channel, 18)
        elif layer_channel%5 == 0:
            images_per_row = np.gcd(layer_channel, 20)
    return images_per_row


def show_featuremap(model_inputs,
                    layer_outputs,
                    layer_channels,
                    layer_names,
                    img_tensor,
                    logdir,
                    verbose=True):
    """
    顯示輸入影像之特徵圖

    parameters
    ----------
    model_inputs   : 模型輸入節點
    layer_outputs  : 特徵圖輸出節點
    layer_channels : 各層特徵圖數量
    layer_names    : 各層之名稱
    img_tensor     : 輸入影像資料，4D Tensor
    logdir         : 保存位置
    verbose        : 直接顯示
    """
    assert img_tensor.shape[0] == 0

    fm_list = []

    activation_model = Model(inputs=model_inputs, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    cnt = 1
    for layer_name, layer_channel, layer_activation in zip(layer_names, layer_channels, activations):
        images_per_row = _cal_images_per_row(layer_channel)
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                            row * size : (row + 1) * size] = channel_image
        
        if verbose:
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                    scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()

        savefig = np.abs(display_grid)/np.abs(np.max(display_grid)) * 255
        
        save_dir = os.path.join(logdir, "feature_maps")
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir)
            
        H, W = savefig.shape
        savefig = savefig.reshape(H, W).astype("uint8")
        image = Image.fromarray(savefig)
        image.save(os.path.join(save_dir, "%d_%s.bmp"%(cnt,layer_name)))

        fm_list.append(image)
        cnt += 1
    
    return fm_list