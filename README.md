# tf2-mycnn

**還在補齊程式中，希望盡量基於相同模式來完成，實際使用可以直接看筆記本的範例程式:smile:**  
**相關文檔、註解會陸續加上，各位看官，如果發現並取用，可能還是需要知道最基本 Keras 用法喔:smile:**  
**因只有我一人，不知啥時可以完成... 最終目標希望能完成類似一個小框架來呼叫分類、分割、物件偵測模型**  

---

這裡統整我曾經實作過的一些經典 CNN 模型架構，並將其寫成類別 `Class` 來呼叫，希望用較少程式碼來完成一些基本功能。  

因為我是從各種論文、書籍、網路資料來進行學習，也參考 TensorFlow、Keras、PyTorch 裡的程式，所以會跟大部分程式碼雷同，純粹是基於將自己學習過的東西進行分享，也歡迎剛接觸 AI 的同學參考。

最後使用 Keras API 作為 Backend，並建構 `KerasModel` 類別物件作為主要核心，實作各種模型架構。  
所有模型都繼承核心 `core` 裡的 `KerasModel` 類，來提供相同的方法 `Method`。  

實現常見的一些經典 CNN Model，

`KerasModel` 中的方法，只是將我自己常用的功能包裝起來，例如：

- 常用參數設定
- 每個檢查點權重保存
- 最佳權重保存
- 訓練過程曲線
- 存/畫出模型結構
- 模型效能評估

> 有些功能是我在 `Keras 1 (Theano)` 、 `Keras 2 (TensorFlow)` 所使用，這邊程式其實是基於當時的類別重新編寫  
> 現在更新到了 `tf.keras`，已經補充很多我當時需要自己寫的功能，也有更多 API 可以運用，減少 `core` 程式量 (~~可說是幾乎沒啥要做了www~~)  

## Feature

- 基於 `tf.keras.Model` 的實例 (instance)，建立核心模型類別 `KerasModel`。  
  ****不採用**** Subclassing Model 的繼承機制來訪問 `tf.keras.Model` 原有屬性/方法。
- 將常見的 Keras 用法寫進 `KerasModel` 類別方法中，只需要設定部分參數即可。
- 藉由繼承 `KerasModel` 來實現各種 CNN Model，同時允許使用 `tf.keras.Model` 原有方法。  
  (必須是未被 `KerasModel` 覆蓋的原方法。如: `compile()`、`fit()`、`evaluate()`、`predict()` . . . 等等。)
- 評估模型性能的工具程式。

> 雖然原有 Keras 的用法已經很簡單，說明文件也很齊全，但還是需要啃許多資料，才能搞懂其機制。  
> 因此，為了一些好同學們能夠直接使用相關程式，將其包裝成新的類別，以此降低使用門檻。  

## Todo

### Classification

- [x] LeNet-5
- [x] AlexNet
- [x] VGG11, VGG13, VGG16, VGG19
- [x] GoogLeNet (Inception v1)
- [x] Inception V3
- [x] ResNet18, ResNet50, ResNet101
- [x] 資料擴增 <font color="red">(尚未驗證)</font>

### Segmentation

- [x] FCN <font color="red">(範例尚未準備)</font>)
- [x] U-Net <font color="red">(範例尚未準備)</font>
- [ ] 分割資料讀取並訓練
- [ ] 分割資料擴增

### Object Detection

- [ ] SSD300
- [ ] RPN
- [ ] RCNN
- [ ] VOC資料讀取並訓練
- [ ] COCO資料讀取並訓練

### Others

- [ ] 利用註解來解釋被我裝進 `KerasModel` 的 Keras API
- [ ] 環境建置說明，我使用的版本號
- [ ] 程式範例說明 (目前有加一些了)
- [ ] 相關文獻整理
