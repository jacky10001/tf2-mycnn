# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


def show_featuremap(model_inputs, layer_outputs, layer_names, img_tensor, logdir, ret_arr):
    fm_list = []

    activation_model = Model(inputs=model_inputs, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    print(len(activations))
    first_layer_activation = activations[0]
    print(first_layer_activation.shape)

    images_per_row = 2

    cnt = 1
    for layer_name, layer_activation in zip(layer_names, activations):
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
    
    return fm_list if ret_arr else None