# -*- coding: utf-8 -*-

import os
import os.path as osp

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_history(history, result_path, mode='a'):
    """
    紀錄Keras模型訓練的過程 (紀錄fit的最終結果)
    
    Keras 模型在 fit 執行結束之後，會返回一個 history 的 dict
    此程式目的為將此訓練過程紀錄保存下來，以方便後續分析查看
    """
    if not(osp.exists(result_path)) or mode != 'a':
        # 檔案不存在 或 覆蓋檔案 時，需要加上標題，
        header=True
    else:
        header=False
    pd.DataFrame(history).to_csv(result_path, encoding='utf-8', mode=mode,
                                 index=False, header=header)


def load_history(result_path):
    """
    讀回Keras模型訓練的過程紀錄檔案
    """
    history = pd.read_csv(result_path)
    return history


def show_history(history, savepath, *keys):
    """
    可視化訓練過程中的loss、accuracy...等指標
    """
    for key in keys:
        plt.plot(history[key], label=key)
    plt.title('Train History - %s'%keys[0])
    plt.xlabel('Epoch')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
    
    if keys[0]:
        plt.savefig(os.path.join(savepath, "history_%s"%keys[0]), bbox_inches="tight")
    plt.show()
    

def find_all_ckpts(log_ckpt_path, only_filename=True) -> list:
    """
    在指定資料夾中，找尋全部儲存的權重路徑 (*依靠副檔名來判斷)
    配合 Keras 的 callbacks 函數 ModelCheckpoint 使用
    """
    checkpoints = next(os.walk(log_ckpt_path))[2]
    if only_filename:
        return checkpoints
    else:
        ckpts_list = []
        for ckpt in checkpoints:
            ext = os.path.splitext(ckpt)[-1]
            if ext == ".h5" or ext == ".h5df":
                ckpts_list.append(os.path.join(log_ckpt_path,ckpt))
        print("Find %d checkpoint"%len(ckpts_list))
        return ckpts_list
    

def find_last_ckpt(log_ckpt_path, log_history_path) -> tuple:
    """
    在指定資料夾中，找尋最後儲存的權重路徑 (*依靠檔名來判斷訓練的epoch)
    配合 Keras 的 callbacks 函數 ModelCheckpoint 使用
    """
    checkpoints = find_all_ckpts(log_ckpt_path)
    if checkpoints:
        print("*"*30)
        print("Loading last checkpoint...")
        last_ckpt_path = os.path.join(log_ckpt_path,checkpoints[-1])
        init_epoch = len(checkpoints)
        
        ## 更新訓練過程紀錄 (手動刪除權重時用)
        hist_csv = load_history(log_history_path)
        save_history(hist_csv[:init_epoch], log_history_path, mode='w+')
        print("Start traing from epoch %d"%init_epoch)
        print("*"*30)
    else:
        print("*"*30+"\nTraining new model...\n"+"*"*30)
        last_ckpt_path = ""
        init_epoch = 0
    return init_epoch, last_ckpt_path
    

def find_best_ckpt(ckpts, pattern=r"weights.(\d{5})-(\d{1,2}.\d{7})", mode="min") -> str:
    """
    在指定資料夾找尋最佳權重結果 (檔名需配合 pattern)
    配合 Keras 的 callbacks 函數 ModelCheckpoint 使用
    """
    if mode.lower() == "min":
        threshold = np.inf
    elif mode.lower() == "max":
        threshold = -np.inf
    best_idx = 0
    for i in ckpts:
        ckpt = re.match(pattern, i)
        ep, val = ckpt.groups()
        ep = int(ep)
        val = float(val)
        if val < threshold and mode.lower() == "min":
            threshold = val
            best_idx = ep
        elif val > threshold and mode.lower() == "max":
            threshold = val
            best_idx = ep
    best_ckpt = ckpts[best_idx]
    print()
    print("best model : {}".format(best_ckpt))
    
    return best_ckpt