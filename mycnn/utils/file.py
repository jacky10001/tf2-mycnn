# -*- coding: utf-8 -*-

import json

    
def save_json(filepath, data_dict):
    """ 
    將 dict 轉成 JSON 檔案，並指定縮排以提升文件可讀性
    
    Note: 可用於保存 Keras 模型的 JSON 檔案
    此文件可搭配 Keras 的 models.model_from_json()
    """
    # solve models.model_from_json()
    if isinstance(data_dict, str):
        data_dict = json.loads(data_dict)
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=4)
    

def load_json(filepath):
    """ 
    將 JSON 轉成 dict 資料結構
    
    Note: 可用於載入 Keras 模型的 JSON 檔案並將其內容轉成 dict 資料結構
    """
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    return data_dict