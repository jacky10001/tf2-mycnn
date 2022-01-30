# -*- coding: utf-8 -*-
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image


def make_voc_segment_dataset(voc_directory: str, save_directory: str):
    flag = False

    ## Set some directory
    JPEGImages_dir = os.path.join(voc_directory, "JPEGImages")
    SegmentationClass_dir = os.path.join(voc_directory, "SegmentationClass")
    ImageSets_dir = os.path.join(voc_directory, "ImageSets", "Segmentation")
    trainval_path = os.path.join(ImageSets_dir, "trainval.txt")

    main_folder = os.path.join(save_directory, "VOCSegmentation")
    train_folder = os.path.join(main_folder, "train")
    train_images_folder = os.path.join(train_folder, "images")
    train_masks_folder = os.path.join(train_folder, "masks")
    train_visualization_folder = os.path.join(train_folder, "visualization")

    ## Check dataset
    check_list = [train_images_folder, train_masks_folder, train_visualization_folder]
    for check_path in check_list:
        if os.path.exists(check_path):
            if not os.listdir(check_path) or len(os.listdir(check_path)) != 2913:
                raise ValueError(f"Detect incomplete data in {check_path}. "
                                 "Please delete all data and unzip again.")
            flag = False
        else:
            flag = True

    print("Make some folders.")
    if not os.path.exists(main_folder):
        os.makedirs(main_folder, exist_ok=True)
    if not os.path.exists(train_images_folder):
        os.makedirs(train_images_folder, exist_ok=True)
    if not os.path.exists(train_masks_folder):
        os.makedirs(train_masks_folder, exist_ok=True)
    if not os.path.exists(train_visualization_folder):
        os.makedirs(train_visualization_folder, exist_ok=True)

    print("Get data list.")
    with open(trainval_path) as f:
        t = f.read().split('\n')[:-1]

    if flag:
        print("Start to make dataset.")
        for name in t:
            ## get file path
            im_path = os.path.join(JPEGImages_dir, name+".jpg")
            gt_path = os.path.join(SegmentationClass_dir, name+".png")

            ## read data
            im = io.imread(im_path)
            vs = Image.open(gt_path)
            gt = Image.open(gt_path)
            gt = np.array(gt)
            gt[gt == 255] = 0

            io.imsave(os.path.join(train_images_folder, os.path.basename(im_path)), im, check_contrast=False)
            io.imsave(os.path.join(train_masks_folder, os.path.basename(gt_path)), gt, check_contrast=False)
            vs.save(os.path.join(train_visualization_folder, os.path.basename(gt_path)))

        print("Finished making dataset.")
    else:
        print("Already made dataset.")