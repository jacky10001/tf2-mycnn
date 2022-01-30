# -*- coding: utf-8 -*-

from .cats_vs_dogs import cats_vs_dogs_from_MSCenter
from .cats_vs_dogs import cats_vs_dogs_by_kaggle_zipfile

from .voc_segment import make_voc_segment_dataset

from .classification import generate_classification_dataset

__all__ = [
    'cats_vs_dogs_from_MSCenter',
    'cats_vs_dogs_by_kaggle_zipfile',
    'make_voc_segment_dataset',
    'generate_classification_dataset'
]