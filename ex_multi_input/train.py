# -*- coding: utf-8 -*-

#%%  引入套件
# Machine learning API
import tensorflow as tf

# Image array processing
import cv2
import numpy as np
import matplotlib.pyplot as plt

# other
import os

# my library
from model import build_model
from data import get_data_list, get_dataset


#%%  參數設置
LOG_DIR = r'.\log'
DIRECTORY_AM = r'D:\jacky\dataset\gd\*\*_A.png'
DIRECTORY_PH = r'D:\jacky\dataset\gd\*\*_P.png'

TRAIN_BATCH_SIZE = 1
TRAIN_EPOCHS = 10

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

os.makedirs(LOG_DIR, exist_ok=True)


#%%  模型架構
print('\ncreate model')

model = build_model(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,1))
model.summary()
tf.keras.utils.plot_model(model, to_file=LOG_DIR+r'\architecture.png', show_shapes=True)


#%%  讀取資料
print('\ncreate dataset .... ')

data_Am, data_Ph, data_Lb = get_data_list(
    DIRECTORY_AM, DIRECTORY_PH, shuffle=True, seed=5)

tra_dataset = get_dataset(
    data_Am[:], data_Ph[:], data_Lb[:],
    TRAIN_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, num_channels=1,
    shuffle=True, seed=10, one_hot=True, num_classes=6)

val_dataset = get_dataset(
    data_Am[:], data_Ph[:], data_Lb[:],
    TRAIN_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, num_channels=1,
    shuffle=False, seed=None, one_hot=True, num_classes=6)

# # here can check dataset
# for element in tra_dataset:
#     print(element)


#%%  訓練模型
print('\ntrain model')

def scheduler(epoch, lr):
    print('check lr:',lr)
    if epoch < 5: return lr
    elif epoch == 50: return lr / 10
    elif epoch == 100: return lr / 10
    else: return lr

def set_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(LOG_DIR+r'\model-{epoch:05d}-{val_loss:.2f}'),
        tf.keras.callbacks.ModelCheckpoint(LOG_DIR+'/best-model', save_best_only=True),
        tf.keras.callbacks.TensorBoard(LOG_DIR+r'\tensorboard'),
        # tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.CSVLogger(LOG_DIR+r'\train_progress.csv')
    ]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

his = model.fit(
    tra_dataset, validation_data=val_dataset,
    epochs=TRAIN_EPOCHS, verbose=1,
    callbacks=set_callbacks()
)

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.legend(['loass','val_loass'], loc='upper right')
plt.savefig(LOG_DIR+r'\train_loss.png')
plt.show()

plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'], loc='lower right')
plt.savefig(LOG_DIR+r'\train_acc.png')
plt.show()


#%%  評估結果
print('\nload model .....')
model = tf.keras.models.load_model(LOG_DIR+'/best-model')

print('\nevaluate model')
eva = model.evaluate(val_dataset, verbose=1)


#%%  預測結果
_idx = 0
Lb = data_Lb[_idx]
Am = tf.io.read_file(data_Am[_idx])
Ph = tf.io.read_file(data_Ph[_idx])

Am = tf.io.decode_png(Am, channels=1)
Ph = tf.io.decode_png(Ph, channels=1)
Am = tf.image.resize(Am, (IMAGE_HEIGHT,IMAGE_WIDTH)).numpy()
Ph = tf.image.resize(Ph, (IMAGE_HEIGHT,IMAGE_WIDTH)).numpy()

Am_tensor = Am.reshape((1,IMAGE_HEIGHT,IMAGE_WIDTH,1))
Ph_tensor = Ph.reshape((1,IMAGE_HEIGHT,IMAGE_WIDTH,1))

print('\npredict model')
print('\nload model .....')
model = tf.keras.models.load_model(LOG_DIR+'/best-model')
pre = model.predict({'xa': Am_tensor, 'xp': Ph_tensor}, verbose=1)

print('\nshow predicted result')
print('ground true:', Lb)
print('    predcit:', pre['yc'][0].argmax(axis=-1))
