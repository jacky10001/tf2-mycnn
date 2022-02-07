# tf2-mycnn

**還在研究 TensorFlow 2 中，不知啥時可以完成...希望盡量以相同概念來完成各種CNN:smile:**  
**相關文檔放在 [GitHub Wiki](https://github.com/jacky10001/tf2-mycnn/wiki)，相關結果可以直接看 Jupyter Notebook 的範例程式**  
**最終目標希望能完成類似一個小框架來呼叫分類、分割、物件偵測模型**  

---

這裡統整我曾經實作過的一些經典 CNN 模型架構，與網路大部分直接寫成 `Function` 不同，我是將其寫成 `Class` 來呼叫。所有 CNN 會繼承 `KerasModel` 類別物件，並通過實例化來共用相關 `Class Method`，用意是將我常用 Keras API 包裝起來，方便重現我之前的一些用法。  

因為我是從各種論文、書籍、網路資料來進行學習，也參考 TensorFlow、Keras、PyTorch 裡的程式，所以會跟大部分程式碼雷同，純粹是基於將自己學習過的東西進行分享，也歡迎剛接觸 AI 的同學參考。

## Todo

### Classification

- [x] LeNet5、AlexNet、VGGNet、Inception、ResNet
- [x] 資料擴增函式 generate_classification_dataset

### Segmentation

- [x] FCN
- [x] U-Net (範例尚未準備)
- [x] 分割資料讀取並訓練
- [ ] 分割資料擴增

### Object Detection

- [ ] SSD300
- [ ] RPN
- [ ] RCNN
- [ ] VOC資料讀取並訓練
- [ ] COCO資料讀取並訓練

### Others

- [x] 多輸入多輸出模型
- [x] 多輸入多輸出的資料讀取

## Refer

- [Dogs VS. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats)
- [Kaggle Cats and Dogs Dataset - Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
