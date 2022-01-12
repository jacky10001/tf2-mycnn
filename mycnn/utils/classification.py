# -*- coding: utf-8 -*-

import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from ..core._file import save_txt, save_json


def export_classification_report(tr: np.ndarray,
                                 pr: np.ndarray,
                                 pb: np.ndarray,
                                 target_names: list,
                                 logpath:str,
                                 verbose=True):
    """
    Export Classification Report with the scikit-learn (a ML API).

    """    
    lb_ids = [n for n in range(len(target_names))]
    
    # 預測結果
    # Name, Geound True PR Probability
    num = len(pb)
    log = []
    for i in tqdm(range(num)):
        log.append([i, tr[i], pr[i]] + [pb[i][n] for n in lb_ids])
    log = pd.DataFrame(log, columns=['Name','Truth','Predict']+target_names)
    log.to_csv(logpath+"/Predicted Results.csv", encoding='utf-8', index=False)
    
    # 混淆矩陣
    cm_report = confusion_matrix(tr, pr)
    if verbose: print(cm_report, "\n")
    cm = pd.DataFrame(cm_report, index=lb_ids, columns=lb_ids)
    cm.to_csv(logpath+"/Confusion Matrix.csv", encoding='utf-8')
    
    # 輸出分類報告
    cls_rep_text = classification_report(tr, pr, target_names=target_names)
    cls_rep_json = classification_report(tr, pr, target_names=target_names, output_dict=True)
    if verbose: print(cls_rep_text, "\n")
    save_txt(logpath+"/Classification Report.txt", cls_rep_text, end='')
    save_json(logpath+"/Classification Report.json", cls_rep_json)
    
    return {
        "confusion_matrix": cm_report, 
        "classification_report": cls_rep_json, 
    }


def plot_confusion_matrix(cm: np.ndarray,
                          target_names: list,
                          logpath: str,
                          title: str='Confusion Matrix'):
    lb_loc = np.arange(len(target_names))
    
    # 圖表設定
    plt.figure(figsize=(10,6), dpi=80)
    
    # 文字設定
    fontsize = 12
    va = 'center'
    ha = 'center'
    rows, cols = np.meshgrid(lb_loc, lb_loc)
    rows = rows.flatten()
    cols = cols.flatten()
    for row, col in zip(rows, cols):
        val = cm[col][row]
        if row == col:
            if val >= 0.89:     color = 'white'
            elif val >= 0.79:   color = 'orange'
            elif val >= 0.49:   color = 'yellow'
            else:               color = 'red'
        else:
            if val >= 0.09:     color = 'red'
            elif val >= 0.009:  color = 'blue'
            else:               color = 'green'
        plt.text(row, col, "%0.2f" %(val,), color=color, fontsize=fontsize, va=va, ha=ha)
    
    # 圖表調整
    lb_ticks = lb_loc + 0.5
    plt.gca().set_xticks(lb_ticks, minor=True)
    plt.gca().set_yticks(lb_ticks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # 填入資料
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.xticks(lb_loc, target_names)
    plt.yticks(lb_loc, target_names)
    plt.ylabel('Ground True label')
    plt.xlabel('Predicted label')
 
    # 顯示資料
    plt.savefig(osp.join(logpath, title))
    plt.show()