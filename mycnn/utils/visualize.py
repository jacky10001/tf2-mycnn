# -*- coding: utf-8 -*-

"""
影像繪製
繪製特徵點
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plt_image_show(image, color=None, title=None, savepath=""):
    plt.imshow(image, cmap=color)
    plt.title(title)
    plt.savefig(savepath)
    plt.show()


def plt_keypoints(im, gd=None, pr=None, savedir=""):
    """
    利用 matplotlib 來標示特徵點
    
    parameters
    ----------
    im      : np.ndarray - 繪製目標影像
    gd      : np.ndarray - 正確標記
    pr      : np.ndarray - 預測的結果
    savedir : str - 設定保存結果的路徑
    """
    
    assert isinstance(im, np.ndarray)
    
    show_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if isinstance(gd, np.ndarray):
        gt_bbox = ( np.min(gd[ : , 0 ]) , np.min(gd[ : , 1 ]) ,
                    np.max(gd[ : , 0 ]) , np.max(gd[ : , 1 ]) )
        x_min, y_min, x_max, y_max = map(int, gt_bbox)
        show_im = cv2.rectangle(show_im, (x_min, y_min), (x_max, y_max), (0,0,255), thickness=1)
    
    if isinstance(pr, np.ndarray):
        pr_bbox = ( np.min(pr[ : , 0 ]) , np.min(pr[ : , 1 ]) ,
                    np.max(pr[ : , 0 ]) , np.max(pr[ : , 1 ]) )
        x_min, y_min, x_max, y_max = map(int, pr_bbox)
        show_im = cv2.rectangle(show_im, (x_min, y_min), (x_max, y_max), (255,0,0), thickness=1)
    
    plt.imshow(show_im)
    if type(gd) == np.ndarray:
        print("\nShow ground true points.")
        plt.scatter(gd[ : , 0 ] , gd[ : , 1 ] , c='yellow' )
    if type(pr) == np.ndarray:
        print("\nShow predicted points.")
        plt.scatter(pr[ : , 0 ] , pr[ : , 1 ] , c='orange' )
    if savedir:
        print("\nSave result to \"%s.\""%savedir)
        plt.savefig(savedir)
    plt.show()