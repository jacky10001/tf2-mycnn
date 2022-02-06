# My CNN

實作的模型都放在這裡，每個模型都會繼承 `core` 裡的 `KerasModel`

## `core`

將常用 Keras API 包裝成類別，將其命名為 `KerasModel`，並實例化物件以呼叫那些方法

## `data`

- 處理相關資料相關程式
- 基於 `tf.data.Dataset` 資料集讀取函式

## `losses`

一些 Keras 沒有的損失函數

## `utils`

一些額外程式都放這裡，評估性能、物件偵測用的程式等等
