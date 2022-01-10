# tf2-mycnn

實現常見的一些經典 CNN Model，並將其寫成類別 (class) 來呼叫，希望用較少程式碼來完成一些基本功能。

> 雖然原有 Keras 的用法已經很簡單，說明文件也很齊全，但還是需要啃許多資料，才能搞懂其機制。  
> 因此，為了一些好同學們能夠直接使用相關程式，將其包裝成新的類別，以此降低使用門檻。  

## Feature

- 基於 `tf.keras.Model` 的實例 (instance)，建立核心模型類別 `KerasModel`。  
  ** **不採用** ** Subclassing Model 的繼承機制來訪問 `tf.keras.Model` 原有屬性/方法。
- 將常見的 Keras 用法寫進 `KerasModel` 類別方法中，只需要設定部分參數即可。
- 藉由繼承 `KerasModel` 來實現各種 CNN Model，同時允許使用 `tf.keras.Model` 原有方法。  
  (必須是未被 `KerasModel` 覆蓋的原方法。如: `compile()`、`fit()`、`evaluate()`、`predict()` . . . 等等。)
- 評估模型性能的工具程式。

## Todo

### Classification

- [x] LeNet-5
- [x] AlexNet
- [ ] VGG16
- [ ] VGG19
- [ ] Inception v1
- [ ] Inception v2
- [ ] Inception v3
- [ ] ResNet-18
- [ ] ResNet-50
- [ ] ResNet-101

### Segmentation

- [ ] FCN
- [ ] UNet

### Object Detection

- [ ] SSD300
- [ ] RCNN

### Others

- [ ] 環境建置說明
- [ ] 程式範例說明
- [ ] 相關文獻整理
