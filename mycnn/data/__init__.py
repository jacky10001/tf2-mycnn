# -*- coding: utf-8 -*-

from .cats_vs_dogs import cats_vs_dogs_from_MSCenter
from .cats_vs_dogs import cats_vs_dogs_by_kaggle_zipfile

from .voc_dataset import download_pascal_voc_dataset
from .voc_segment import make_voc_segment_dataset

from .classification import generate_classification_dataset
from .segmentation import generate_segmentation_dataset

__all__ = [
    'cats_vs_dogs_from_MSCenter',
    'cats_vs_dogs_by_kaggle_zipfile',
    'download_pascal_voc_dataset',
    'make_voc_segment_dataset',
    'generate_classification_dataset',
    'generate_segmentation_dataset'
]