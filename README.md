# tf2-mycnn

**還在研究中，希望盡量基於相同模式來完成，實際使用可以直接看筆記本的範例程式:smile:**  
**相關文檔放在 Github Wiki，等大致工作都完成後，再來把說明文件完成:smile:**  
**因只有我一人，不知啥時可以完成... 最終目標希望能完成類似一個小框架來呼叫分類、分割、物件偵測模型**  

---

這裡統整我曾經實作過的一些經典 CNN 模型架構，與網路大部分直接寫成函式程式不同，  
我是將其寫成類別 `Class` 來呼叫 我自己寫的 `KerasModel` 類別物件，  
並實例化這個物件來實作各種 CNN 模型，用意是將常用 Keras API 包裝成一些方法。  

因為我是從各種論文、書籍、網路資料來進行學習，也參考 TensorFlow、Keras、PyTorch 裡的程式，  
所以會跟大部分程式碼雷同，純粹是基於將自己學習過的東西進行分享，也歡迎剛接觸 AI 的同學參考。

我的開發環境整理在 [Github Wiki](https://github.com/jacky10001/tf2-mycnn/wiki/%E9%96%8B%E7%99%BC%E6%96%87%E6%AA%94-%E7%92%B0%E5%A2%83%E8%88%87%E4%BE%9D%E8%B3%B4%E5%A5%97%E4%BB%B6)

## Todo

### Classification

- [x] LeNet5、AlexNet、VGGNet、Inception、ResNet
- [x] 資料擴增函式 generate_classification_dataset
- [ ] 測試我的擴增函式對於模型是否有正向影響 (*目前結果奇差無比........)

### Segmentation

- [x] FCN (範例尚未準備)
- [x] U-Net (範例尚未準備)
- [ ] 分割資料讀取並訓練
- [ ] 分割資料擴增

### Object Detection

- [ ] SSD300
- [ ] RPN
- [ ] RCNN
- [ ] VOC資料讀取並訓練
- [ ] COCO資料讀取並訓練

## Refer

- [Dogs VS. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats)
- [Kaggle Cats and Dogs Dataset - Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
