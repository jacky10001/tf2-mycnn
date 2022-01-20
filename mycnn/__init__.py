# -*- coding: utf-8 -*-

from . import losses
from . import data
from . import utils

from .lenet import LeNet5
from .alexnet import AlexNet
from .vgg import VGG11
from .vgg import VGG13
from .vgg import VGG16
from .vgg import VGG19
from .resnet import ResNet18
from .resnet import ResNet50
from .resnet import ResNet101
from .googlenet import GoogleNet
from .inception_v3 import InceptionV3

__version__ = "v0.22.01.20"
__maintainer__ = "Jacky"