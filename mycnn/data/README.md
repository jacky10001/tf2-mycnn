# `data`

## cats_vs_dogs.py

自動下載貓狗資料集 (from Microsoft Download Center)
自動整理貓狗資料集的檔案結構 (from Kaggle)

## classification.py

我基於 `tf.data.Dataset` 實現的資料讀取函式，除了一般的影像讀取外，也包含資料擴增的功能  
TensorFlow 2 不知幾版後，有實現一個 API 叫做 `tf.keras.preprocessing.image_dataset_from_directory`  
基本上我的實現跟這個很像，但是人家沒進行資料擴增，訓練結果就已經比較好，真心不知道差在哪裡...  
我推測這個可以搭配 Keras 的預處理層，但是我還是希望 CNN 的 Model 不要添加這些額外層比較好理解...  
把資料處理跟模型分開來，但是目前的結果非但沒提升，反而變慘了哈哈哈。
