# -*- coding: utf-8 -*-

"""
基於 Keras API 來實現特殊 activation function
"""

from keras import backend as K


def custom_relu(x):
    """ Applies the rectified linear unit activation function.

    K.relu(x, alpha=0.0, max_value=None, threshold=0)
    參數說明:
        x: 輸入變數(variable)或張量(tensor)
        alpha: 控制低於閾值(threshold)的值的斜率(slope)的浮點數(float)。
        max_value: 設置飽和閾值(saturation threshold)的浮點數。(函數將返回的最大值)
        threshold: 一個浮點數，給出激活函數的閾值，低於該值將被抑製(damped)或設置為零。
    """
    return K.relu(x, max_value=1)


def custom_sigmoid(x):
    """ Applies the sigmoid activation function. 

    Formula:
        sigmoid(x) = 1 / (1 + exp(-x))
    
    對於小值 (<-5)，sigmoid 返回一個接近於零的值，
    對於大值 (>5)，函數的結果接近於 1。
    
    當第二個元素假設為零，Sigmoid 相當於一個 2 元素的 Softmax。
    
    sigmoid 函數總是返回一個介於 0 和 1 之間的值。
    """
    return 1 / (1 + K.exp(-x))


def custom_softplus(x):
    """ Applies the softplus activation function. 

    Formula:
        softplus(x) = log(exp(x) + 1)
    """
    return K.log(K.exp(x) + 1)


if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    ## Demo: sigmoid
    a = np.linspace(-1,1,65)*10
    b = 1 / (1 + np.exp(-a))
    plt.plot(a,b)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.show()
    ## Demo: softplus
    a = np.linspace(-1,1,65)
    b = np.log(np.exp(a) + 1)
    plt.plot(a,b)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.show()
    ## Demo: exp
    a = np.linspace(-1,1,65)
    b = np.exp(a)
    plt.plot(a,b)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.show()