# -*- coding: utf-8 -*-

from . import core
from . import losses
from . import data
from . import utils

# classification
from .lenet import LeNet5
from .alexnet import AlexNet
from .vgg import VGG11
from .vgg import VGG13
from .vgg import VGG16
from .vgg import VGG19
from .resnet import ResNet18
from .resnet import ResNet50
from .resnet import ResNet101
from .googlenet import GoogLeNet
from .inception_v3 import InceptionV3

# segmentation
from .fcn import FCN32
from .fcn import FCN16
from .fcn import FCN8
from .fcn import FCN32_KERAS
from .fcn import FCN16_KERAS
from .fcn import FCN8_KERAS
from .unet import UNet

__version__ = "v0.22.02.11"
__maintainer__ = "Jacky"