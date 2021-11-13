import torch
import os
import glob
import cv2
import numpy as np


def img2tensor(img_arr):
    '''float64 ndarray (H,W,3) ---> float32 torch tensor (1,3,H,W)'''
    img_arr = img_arr.astype(np.float32)
    img_arr = img_arr.transpose(2, 0, 1)  # channel first
    img_arr = img_arr[np.newaxis, :, :, :]
    init_tensor = torch.from_numpy(img_arr)  # (1,3,H,W)
    return init_tensor


def normalize(im_tensor):
    '''(0,255) ---> (-1,1)'''
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor


def tensor2img(tensor):
    '''(0,255) tensor ---> (0,255) img'''
    '''(1,3,H,W) ---> (H,W,3)'''
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    img = tensor.cpu().numpy().clip(0, 255).astype(np.uint8)
    cv2.imwrite('img.jpg', img)
    return img

def tensor2imgs(tensor):
    '''(0,255) tensor ---> (0,255) img'''
    '''(1,3,H,W) ---> (H,W,3)'''
    # tensor = tensor.permute(0, 2, 3, 1)
    imgs = tensor.cpu().numpy().clip(0, 255).astype(np.uint8)
    return imgs
