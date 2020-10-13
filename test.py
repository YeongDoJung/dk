from models.densenet_models import DenseNet
import torch
import torchvision
from torchsummary import summary
import torch.nn as nn
from utils.utils import AverageMeter
import pandas as pd
from models.densenet_models import DenseNet
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt


rawpath = '/media/disk/data/dk_rawfile/proc20201005_174709_CHEST_PA_512_512_16bit.raw'
img = np.fromfile(rawpath, dtype=uint16)

# plt.imshow(image_raw)
  
    



    

