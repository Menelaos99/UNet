import os
import numpy as np
import json
from skimage.draw import polygon
from PIL import Image
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import time
import copy
from torchvision.io import read_image
from torchmetrics import Dice
import cv2
from torchvision.io import read_image



class CarvanaDataset():
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
       
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image_batch, mask_batch = augment_crop(img_path, mask_path)

        return image_batch, mask_batch

TRAIN_IMG_DIR = r"/Users/menelaos/Desktop/U-Nets/dataset/train_images"
TRAIN_MASK_DIR = r"/Users/menelaos/Desktop/U-Nets/dataset/train_masks"
VAL_IMG_DIR = r"/Users/menelaos/Desktop/U-Nets/dataset/val_images"
VAL_MASK_DIR = r"/Users/menelaos/Desktop/U-Nets/dataset/val_masks"
TEST_IMG_DIR = r"/Users/menelaos/Desktop/U-Nets/dataset/test_images"
TEST_MASK_DIR = r"/Users/menelaos/Desktop/U-Nets/dataset/test_masks"

IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
EPOCHS = 50
LEARNING_RATE = 1e-5

def elastic_transform(image, alpha, sigma, random_state=None):

        if random_state is None:
            random_state = np.random.RandomState(None)
    
        shape_size = image.shape[:2]
    
        grid_scale = 4
        alpha //= grid_scale  
        sigma //= grid_scale  
        grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)
    
        blur_size = int(4 * sigma) | 1
        rand_x = cv2.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        rand_y = cv2.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        if grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])
    
        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        grid_x = (grid_x + rand_x).astype(np.float32)
        grid_y = (grid_y + rand_y).astype(np.float32)
    
        distorted_img = cv2.remap(image, grid_x, grid_y,
            borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)
    
        return distorted_img

def augment_crop(files1, files2, dir1, dir2):
        for file1 in files1:
            for file2 in files2:
                if file1 == file2:
                    img_arr1 = cv2.imread(dir1 + file1, -1)
                    img_arr2 = cv2.imread(dir2 + file2, -1)   
    
                    img_arr2 = np.delete(img_arr2, 0, 0)
                    img_arr2 = np.delete(img_arr2, 0, 1)
    
                    im_merge = np.concatenate((img_arr1[...,None], img_arr2[...,None]), axis=2)
                    
                    #augment
                    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 9, im_merge.shape[1] * 0.09)

                    img1_deformed = im_merge_t[...,0]
                    img2_deformed = im_merge_t[...,1]

                    #crop
                    img1_deformed = img1_deformed[50:450, 50:650]
                    img2_deformed = img2_deformed[50:450, 50:650]
                    
                    new_image = torch.tensor(img1_deformed, dtype=torch.float32).permute(2, 0, 1)
                    new_mask = torch.tensor(img2_deformed, dtype=torch.float32).unsqueeze(0)

                    return new_image, new_mask

train_ds = CarvanaDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform= augment_crop)
val_ds = CarvanaDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform= augment_crop)
test_ds = CarvanaDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transform= augment_crop)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=True)

        
# image = Image.open(img_path)
# image = np.array(image.convert('RGB'))
# mask = Image.open(mask_path)
# mask = np.array(mask.convert('L'))

# image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
# mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

image_dir = r"/Users/menelaos/Desktop/U-Nets/dataset/train_images"
mask_dir = r"/Users/menelaos/Desktop/U-Nets/dataset/train_masks"

image_dir = image_dir
mask_dir = mask_dir

img_path = os.listdir(image_dir)
mask_path = os.listdir(mask_dir)
        
image_batch, mask_batch = augment_crop(img_path, mask_path)

a, b = augment_crop(img_path)
        