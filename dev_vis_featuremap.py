# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image

img_path = r'D:\YangJie\proj_ai\Particle\database\case03\data\09001_01_O.bmp'
MODEL_DIR = config.MODEL_DIR
INPUT_SIZE = config.INPUT_SIZE
#%% preporcessing image to 4D tensor
assert os.path.exists(img_path) == True
img = image.load_img(img_path, target_size=(256,256), grayscale=True)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0).astype('float32')
print(img_tensor.shape)
plt.imshow(img_tensor[0,:,:,0],cmap='gray')
plt.xticks(range(0,256,32))
plt.yticks(range(0,256,32))
plt.grid(color='r', linestyle='-', linewidth=1)
plt.show()
#%% Load previous training weight
assert os.path.exists(MODEL_DIR) == True
print('Load previous training.\n')
model = DNN(INPUT_SIZE, MODEL_DIR)

layer_outputs = [layer.output for layer in model.layers[1:31]]
for op in layer_outputs: 
    print(op)

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
print(len(activations))
first_layer_activation = activations[0]
print(first_layer_activation.shape)
#%% visualization only one feature
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
#%% visualization every channel feature
layer_names = []

for layer in model.layers[1:31]:
    layer_names.append(layer.name)

images_per_row = 8

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
    
    save_dir = r'output\visualization'
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
        
    H, W = savefig.shape
    savefig = savefig.reshape(H, W, 1)
    image.save_img(r'%s\%s.png'%(save_dir,layer_name), savefig)