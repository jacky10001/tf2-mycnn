# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from collections import OrderedDict

img_path = r'D:\YangJie\proj_ai\Particle\database\case03\data\09001_01_O.bmp'
MODEL_DIR = config.MODEL_DIR
INPUT_SIZE = config.INPUT_SIZE
layerDict = OrderedDict(
        [('b1_7_conv',32),('b2_7_conv',64),('b3_7_conv',128)
])
#%%
assert os.path.exists(MODEL_DIR) == True
print('Load previous training.\n')
model = DNN(INPUT_SIZE,MODEL_DIR)
#%%
## 建立將圖像張量轉換為可用的影像格式的自訂函式
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)				# 1. 張量正規化：以 0 為中心, 確保 std 為 0.1 
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1) # 修正成 [0, 1], 即 0-1 之間 
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

## 建立視覺化過濾器的函式
def generate_pattern(layer_name, filter_index, size=256):
	layer_output = model.get_layer(layer_name).output # 取得指定層的輸出張量
	loss = K.mean(layer_output[:, :, :, filter_index]) # 1. 取得指定過濾器的輸出張量, 並以最大化此張量的均值做為損失

	grads = K.gradients(loss, model.input)[0] # 根據此損失計算輸入影像的梯度

	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # 標準化技巧：梯度標準化

	iterate = K.function([model.input], [loss, grads]) # 2.建立 Keras function 來針對給定的輸入影像回傳損失和梯度

	input_img_data = np.random.random((1, size, size, 1)) * 20 + 128. # 3. 從帶有雜訊的灰階影像開始

	step = 1.
	for i in range(40): # 執行梯度上升 40 步
		loss_value, grads_value = iterate([input_img_data]) # 4. 針對給定的輸入影像回傳損失和梯度
		input_img_data += grads_value * step

	img = input_img_data[0]
	return deprocess_image(img)	  # 進行圖像後處理後回傳

## 產生一層中所有的過濾器響應 pattern
n_rows = 8
for layer_name, n_filters in layerDict.items():
    size = 64
    margin = 5
    n_cols = n_filters // n_rows
    
    # 1. 用於儲存結果的空(黑色)影像
    results = np.zeros((n_cols*size+(n_cols-1)*margin, n_rows*size+(n_rows-1)*margin, 1))

    for col in range(n_cols):  # ← 迭代產生網格的行
        for row in range(n_rows):  # ←迭代產生網格的列
            # 在 layer_name 中產生過濾器 col +(j * 8) 的 pattern
            filter_img = generate_pattern(layer_name, col*n_rows+row, size=size)

            # 將結果放在結果網格的方形(i, j)中
            horizontal_start = col*size + col*margin
            horizontal_end = horizontal_start + size
            vertical_start = row*size + row*margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # 顯示網格結
    plt.figure(figsize=(14, 14))
    plt.imshow(results[:,:,0])
    plt.show()
    
    save_dir = r'output\visualization'
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    image.save_img(r'%s\%s.png'%(save_dir,layer_name+'_filter.png'), results)