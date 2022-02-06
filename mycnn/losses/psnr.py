# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf


def psnr(y_true, y_pred):
    return -10*K.log(K.mean(K.flatten((y_true - y_pred))**2))/K.log(10)