# -*- coding: utf-8 -*-
import os
import shutil
import zipfile
import requests
import cv2


def cats_vs_dogs_from_MSCenter(dataset_path):
    # 自動設定相關路徑
    main_folder = os.path.join(dataset_path, "DogsVsCats")
    temp_folder = os.path.join(main_folder, ".~temp")
    train_folder = os.path.join(main_folder, "train")
    original_zip_file = os.path.join(main_folder, "dogs-vs-cats.zip")
    labeldirs = ["Cats", "Dogs"]
    validnums = [12476, 12470]
    flag = False

    if not os.path.exists(main_folder):
        os.makedirs(main_folder, exist_ok=True)

    # 從 MS Center 下載資料集
    if not os.path.exists(original_zip_file):
        url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
        r = requests.get(url, allow_redirects=True)
        open(original_zip_file, "wb").write(r.content)
    else:
        print("Already download zip file.")

    for labldir, validnum in zip(labeldirs, validnums):
        check_path = os.path.join(train_folder, labldir)
        if os.path.exists(check_path):
            if not os.listdir(check_path) or len(os.listdir(check_path)) != validnum:
                raise ValueError(f"Detect incomplete data in {check_path}. "
                                "Please delete all data and unzip again.")
            flag = False
        else:
            flag = True

    if flag:
        print("Beginning make dataset.")

        # 解壓縮檔案至暫存資料夾
        with zipfile.ZipFile(original_zip_file, 'r') as zip_ref:
            print(f"Unzipping {original_zip_file} ...")
            os.makedirs(temp_folder, exist_ok=True)
            zip_ref.extractall(temp_folder)

        # 檢查路徑下是否建立資料夾類別
        # 如果尚未建立則會自動搬移資料
        # 若成功偵測到資料夾類別並正確，將不進行任何動作
        # 若偵測資料夾類別但有異常，則會報錯，需重新建立
        for labldir in labeldirs:
            newdir = os.path.join(train_folder, labldir)
            if not os.path.exists(newdir):
                os.makedirs(newdir, exist_ok=True)

        # 移動資料
        cnt_invalid = 0
        print("Moving data to label directorys ...")
        temp_cat_folder = os.path.join(temp_folder, "PetImages", "Cat")
        temp_dog_folder = os.path.join(temp_folder, "PetImages", "Dog")
        for file in os.listdir(temp_cat_folder):
            if file == "Thumbs.db":
                continue
            src = os.path.join(temp_cat_folder, file)
            dst = os.path.join(train_folder, labeldirs[0], file)
            try:
                valid_img = cv2.imread(src)
                valid_img.shape
                shutil.move(src, dst)
            except:
                cnt_invalid += 1
        for file in os.listdir(temp_dog_folder):
            if file == "Thumbs.db":
                continue
            src = os.path.join(temp_dog_folder, file)
            dst = os.path.join(train_folder, labeldirs[1], file)
            try:
                valid_img = cv2.imread(src)
                valid_img.shape
                shutil.move(src, dst)
            except:
                cnt_invalid += 1

        # 移除暫存檔案
        if os.path.exists(temp_folder):
            print(f"Removing {temp_folder} ...")
            shutil.rmtree(temp_folder)
        
        print(f"Detect {cnt_invalid} invalid image.")
        print("Making sucessfully.")
    else:
        print("Already make dataset.")


def cats_vs_dogs_by_kaggle_zipfile(dataset_path):
    # 自動設定相關路徑
    main_folder = os.path.join(dataset_path, "DogsVsCats")
    temp_folder = os.path.join(main_folder, ".~temp")
    train_folder = os.path.join(main_folder, "train")
    original_zip_file = os.path.join(dataset_path, "dogs-vs-cats.zip")
    train_zip_file = os.path.join(temp_folder, "train.zip")
    labeldirs = ["Cats", "Dogs"]
    flag = False

    # 檢查壓縮檔
    if not os.path.exists(original_zip_file):
        raise ValueError(f"Not find original zip file from {original_zip_file}. "
                         "Please check your zip file path or you can download from the Kaggle. "
                         "Please find URL https://www.kaggle.com/c/dogs-vs-cats/data")

    for labldir in labeldirs:
        check_path = os.path.join(train_folder, labldir)
        if os.path.exists(check_path):
            if not os.listdir(check_path) or len(os.listdir(check_path)) != 12500:
                raise ValueError(f"Detect incomplete data in {check_path}. "
                                "Please delete all data and unzip again.")
            flag = False
        else:
            flag = True

    if flag:
        print("Beginning make dataset.")

        # 解壓縮檔案至暫存資料夾
        with zipfile.ZipFile(original_zip_file, 'r') as zip_ref:
            print(f"Unzipping {original_zip_file} ...")
            os.makedirs(temp_folder, exist_ok=True)
            zip_ref.extractall(temp_folder)

        # 從暫存資料夾取得訓練資料
        with zipfile.ZipFile(train_zip_file, 'r') as zip_ref:
            print(f"Unzipping {train_zip_file} ...")
            zip_ref.extractall(main_folder)

        # 檢查路徑下是否建立資料夾類別
        # 如果尚未建立則會自動搬移資料
        # 若成功偵測到資料夾類別並正確，將不進行任何動作
        # 若偵測資料夾類別但有異常，則會報錯，需重新建立
        for labldir in labeldirs:
            newdir = os.path.join(train_folder, labldir)
            if not os.path.exists(newdir):
                os.makedirs(newdir, exist_ok=True)

        # 移動資料
        print("Moving data to label directorys ...")
        for file in os.listdir(train_folder):
            if file.startswith('cat'):
                src = os.path.join(train_folder, file)
                dst = os.path.join(train_folder, labeldirs[0], file)
                shutil.move(src, dst)
            elif file.startswith('dog'):
                src = os.path.join(train_folder, file)
                dst = os.path.join(train_folder, labeldirs[1], file)
                shutil.move(src, dst)

        # 移除暫存檔案
        if os.path.exists(temp_folder):
            print(f"Removing {temp_folder} ...")
            shutil.rmtree(temp_folder)
        
        print("Making sucessfully.")
    else:
        print("Already make dataset.")