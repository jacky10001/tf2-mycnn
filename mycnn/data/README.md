# `data`

## classification.py

我基於 `tf.data.Dataset` 實現的資料讀取函式，除了一般的影像讀取外，也包含資料擴增的功能  
TensorFlow 2 不知幾版後，有實現一個 API 叫做 `tf.keras.preprocessing.image_dataset_from_directory`  
我推測這個可以搭配 Keras 的預處理層，但是我還是希望 CNN 的 Model 不要添加這些額外層比較好理解...  
把資料處理跟模型分開來，但是目前使用資料擴增後，準確度不但沒提升，反而變慘了哈哈哈。

## cats_vs_dogs.py

自動下載貓狗資料集 (from Microsoft Download Center)  
自動整理貓狗資料集的檔案結構 (from Kaggle)

## voc_segment.py

處理 voc 的分割資料，並儲存至指定資料夾中，方便之後直接使用 `tf.data.Dataset` 進行讀取
