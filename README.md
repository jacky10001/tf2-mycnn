# tf2-mycnn

**還在補齊程式中，希望盡量基於相同模式來完成，實際使用可以直接看筆記本的範例程式:smile:**  
**相關文檔、註解會陸續加上，各位看官，如果發現並取用，可能還是需要知道最基本 Keras 用法喔:smile:**  
**因只有我一人，不知啥時可以完成... 最終目標希望能完成類似一個小框架來呼叫分類、分割、物件偵測模型**  

---

這裡統整我曾經實作過的一些經典 CNN 模型架構，並將其寫成類別 `Class` 來呼叫，希望用較少程式碼來完成一些基本功能。  

因為我是從各種論文、書籍、網路資料來進行學習，也參考 TensorFlow、Keras、PyTorch 裡的程式，所以會跟大部分程式碼雷同，純粹是基於將自己學習過的東西進行分享，也歡迎剛接觸 AI 的同學參考。

最後使用 Keras API 作為 Backend，並建構 `KerasModel` 類別物件作為主要核心，實作各種模型架構。  
所有模型都繼承核心 `core` 裡的 `KerasModel` 類，來提供相同的方法 `Method`。  

## Todo

### Classification

- [x] LeNet-5
- [x] AlexNet
- [x] VGG11, VGG13, VGG16, VGG19
- [x] GoogLeNet (Inception v1)
- [x] Inception V3
- [x] ResNet18, ResNet50, ResNet101
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

## Env

### Hardware

- Intel(R) Core(TM) i7-7700HQ CPU @ 2.8GHz
- 24 GB RAM
- NVIDA GTX 1060 6GB

### Software

- Windows 10 21H2
- Python 3.7.10 (Anaconda3)
- Cuda 11.0 + cudnn

### Package

- tensorflow==2.4.1
- scikit-learn==0.24.2
- numpy==1.19.5
- scipy==1.6.3
- h5py==2.10.0
- pydot==1.2.2
- scikit-image==0.18.1
- opencv==4.5.1.48
- Pillow==8.2.0
- pandas==1.2.4
- natsort==8.0.2
- tqdm==4.60.0
