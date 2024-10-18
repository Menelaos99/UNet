import os
import numpy as np
import json
from skimage.draw import polygon
from PIL import Image
import matplotlib.pyplot as plt
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import time
import copy
import random
import cv2
from tqdm import tqdm
import json

#read json file with validation images 
with open('val.json', 'r') as f:
    data = json.load(f)

shape = (521, 705, 3)

def get_image_names(image_dir):
    image_name_list = []
    files = os.listdir(image_dir)
    for file in files:
        name, extension = os.path.splitext(file)
        if extension == '.tif':
            image_name_list.append(name)

