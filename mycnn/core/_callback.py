# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from tensorflow.keras import callbacks


class PredictDataInTraining(callbacks.Callback):
    def __init__(self, logdir, im_tensor):
        super().__init__()
        self.logdir = logdir
        self.im_tensor = im_tensor
        assert os.path.exists(self.img_path) == True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        pr = self.model.predict(self.im_tensor, verbose=0)
        print(epoch, pr[0,...])

        #TODO: write a log file in every epoch.