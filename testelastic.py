from PIL import Image
import copy
import os
import numpy as np, imageio, elasticdeform as ed
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

img_dir = '/Users/menelaos/Desktop/U-Nets/images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif'
img_dir1 = '/Users/menelaos/Desktop/U-Nets/dataset/val_images/'
img_dir2 = '/Users/menelaos/Desktop/U-Nets/dataset/val_masks/'

save_dir1 = '/Users/menelaos/Desktop/U-Nets/dataset/a&c_val_images'
save_dir2 = '/Users/menelaos/Desktop/U-Nets/dataset/a&c_val_masks'

files1 = os.listdir(img_dir1)
files2 = os.listdir(img_dir2)

def elastic_transform(image, alpha, sigma, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape_size = image.shape[:2]

    # Downscaling the random grid and then upsizing post filter
    # improves performance. Approx 3x for scale of 4, diminishing returns after.
    grid_scale = 4
    alpha //= grid_scale  # Does scaling these make sense? seems to provide
    sigma //= grid_scale  # more similar end result when scaling grid used.
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

def augment_crop(folder1, folder2, save_dir1, save_dir2, start, end):
    for file1 in folder1:
        for file2 in folder2:
            if file1 == file2:
                img_arr1 = cv2.imread(img_dir1 + file1, -1)
                img_arr2 = cv2.imread(img_dir2 + file2, -1)   

                img_arr2 = np.delete(img_arr2, 0, 0)
                img_arr2 = np.delete(img_arr2, 0, 1)
 
                im_merge = np.concatenate((img_arr1[...,None], img_arr2[...,None]), axis=2)

                i = start
                sub = i - 1
                while i < end:
                    #augment
                    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * i, im_merge.shape[1] * i * 0.01)

                    img1_deformed = im_merge_t[...,0]
                    img2_deformed = im_merge_t[...,1]

                    #crop
                    img1_deformed = img1_deformed[50:450, 50:650]
                    img2_deformed = img2_deformed[50:450, 50:650]

                    img1_deformed = Image.fromarray(img1_deformed)
                    img2_deformed = Image.fromarray(img2_deformed)

                    image_num = i - sub
                    file1_split = file1.split('.png')
                    file2_split = file2.split('.png')

                    img1_deformed.save(os.path.join(save_dir1, rf'{file1_split[0]}_a&c_{image_num}.png'))
                    img2_deformed.save(os.path.join(save_dir2, rf'{file2_split[0]}_a&c_{image_num}.png'))

                    i += 1
augment_crop(files1, files2, save_dir1, save_dir2, 8, 13)
