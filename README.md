# tf2-mycnn

實現常見的一些經典 CNN Model，並將其寫成類別 `Class` 來呼叫，希望用較少程式碼來完成一些基本功能。  
所有模型都繼承核心 `core` 裡的 `KerasModel` 類，來提供相同的方法 `Method`。  

`KerasModel` 中的方法，只是將我自己常用的功能包裝起來:smile:，例如：

- 每個檢查點權重保存
- 最佳權重保存
- 訓練過程曲線
- 存/畫出模型結構
- 模型效能評估

> 雖然原有 Keras 的用法已經很簡單，說明文件也很齊全，但還是需要啃許多資料，才能搞懂其機制。  
> 因此，為了一些好同學們能夠直接使用相關程式，將其包裝成新的類別，以此降低使用門檻。  

> 有些功能是我在 `Keras 1 (Theano)` 、 `Keras 2 (TensorFlow)` 所使用，這邊程式其實是基於當時的類別重新編寫  
> 也因為現在更新到了 `tf.keras`，已經補充很多我當時需要自己寫的功能，也有更多 API 可以運用，減少 `core` 程式量  
> 

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
- [x] VGG11, VGG13, VGG16, VGG19
- [ ] Inception v1, v2, v3, v4
- [x] ResNet18, ResNet50, ResNet101

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
