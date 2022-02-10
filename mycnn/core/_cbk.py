# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from tensorflow.keras import callbacks


class saveEpochResult(callbacks.Callback):
    def __init__(self, log_dir, img_path):
        super().__init__()
        self.log_dir = log_dir
        self.img_path = img_path
        assert os.path.exists(self.img_path) == True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        x = cv2.imread(self.img_path, 0)
        pr = self.model.predict(x.reshape(1,256,256,1), verbose=0)
        pr_img = pr.reshape(256,256,2)
        
        os.makedirs(self.log_dir, exist_ok=True)
        for ch, name in enumerate(['L','A']):
            path = os.path.join(self.log_dir,"pr%s_e{epoch:04d}.bmp"%name)
            path = path.format(epoch=epoch+1, **logs)
            cv2.imwrite(path,pr_img[:,:,ch])