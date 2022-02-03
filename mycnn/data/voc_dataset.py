# -*- coding: utf-8 -*-
import os
import shutil
import tarfile
import requests
import tensorflow as tf
import cv2


def download_pascal_voc_dataset(dataset_path):
    # 自動設定相關路徑
    main_folder = os.path.join(dataset_path, "VOC")
    untar_folder = os.path.join(main_folder, "VOCdevkit")
    tar_file = os.path.join(main_folder, "VOCtrainval_11-May-2012.tar")

    if not os.path.exists(main_folder):
        os.makedirs(main_folder, exist_ok=True)

    # 從 MS Center 下載資料集
    if not os.path.exists(tar_file):
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        r = requests.get(url, allow_redirects=True)
        open(tar_file, "wb").write(r.content)
    else:
        print("Already download tar file.")
    
    if not os.path.exists(untar_folder):
        # 解壓縮檔案至暫存資料夾
        with tarfile.TarFile(tar_file, 'r') as tar_ref:
            print(f"Untarring {tar_file} ...")
            os.makedirs(main_folder, exist_ok=True)
            tar_ref.extractall(main_folder)
    else:
        print("Already Uutar tar file.")