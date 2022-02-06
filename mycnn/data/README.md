# `data`

## classification.py

我基於 `tf.data.Dataset` 實現的資料讀取函式，除了一般的影像讀取外，也包含資料擴增的功能  
TensorFlow 2 不知幾版後，有實現一個 API 叫做 `tf.keras.preprocessing.image_dataset_from_directory`  
我推測這個可以搭配 Keras 的預處理層，但是我還是希望 CNN 的 Model 不要添加這些額外層比較好理解...  

## segmentation.py

基於 `tf.data.Dataset` 實現的分割資料處理函式，需要注意的是標記資料 (mask)  
必須先行處理成灰階格式 0 ~ 255，包含背景最多只能有 256 個類別，0 通常設為背景  
此模組與 classification.py 幾乎相同形式進行處理，主要差異在於標記資料的處理不同

## cats_vs_dogs.py

自動下載貓狗資料集 (from Microsoft Download Center)  
自動整理貓狗資料集的檔案結構 (from Kaggle)

## voc_segment.py

處理 voc 的分割資料，並儲存至指定資料夾中，方便之後直接使用 `tf.data.Dataset` 進行讀取，  
主要是將標記資料 (mask) 處理成灰階圖片，並以 0 ~ 255 當作類別編碼，一般將 0 作為背景值，  
因此分割的標記資料的類別數量最多只有 256 類 (含背景)
