import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import color
from PIL import Image, ImageMorph
import torch
from skimage.morphology import dilation, square
import os

class TrainDataset(Dataset):
    def __init__(self, data_file_path, K=64, device = None):    
        self.data_root = data_file_path
        self.k = K
        with open(os.path.join(self.data_root, 'train.txt'), 'r') as file:
            self.data = [line.rstrip('\n') for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pic_name = self.data[idx]
        
        shadowfree_img_path = os.path.join(self.data_root, 'shadowfree_imgs', pic_name)
        object_mask_path = os.path.join(self.data_root, 'object_masks', pic_name)
        shadow_mask_path = os.path.join(self.data_root, 'shadow_masks', pic_name)
        shadow_img_path = os.path.join(self.data_root, 'shadow_imgs', pic_name)
        background_object_mask_path = os.path.join(self.data_root, 'background_object_masks', pic_name)
        background_shadow_mask_path = os.path.join(self.data_root, 'background_shadow_masks', pic_name)
        prompt = ''
        
        width, height = 512, 512
        width_mask, height_mask = 64, 64

        shadowfree_img = cv2.imread(shadowfree_img_path)
        shadowfree_img = cv2.resize(shadowfree_img, (width, height))
        object_mask = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask = cv2.resize(object_mask, (width, height))
        background_object_mask = cv2.imread(background_object_mask_path, cv2.IMREAD_GRAYSCALE)
        background_object_mask = cv2.resize(background_object_mask, (width, height))
        background_shadow_mask = cv2.imread(background_shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        background_shadow_mask = cv2.resize(background_shadow_mask, (width, height))
        shadow_img = cv2.imread(shadow_img_path)
        shadow_img = cv2.resize(shadow_img, (width, height))
        shadow_mask = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        shadow_mask = cv2.resize(shadow_mask, (width, height))
        _, fg_instance_thresh = cv2.threshold(object_mask, 128, 255, cv2.THRESH_BINARY)
        contours_instance, _ = cv2.findContours(fg_instance_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        merged_contour_points_instance = np.concatenate(contours_instance)
        rect_instance = cv2.minAreaRect(merged_contour_points_instance)
        (x, y), (w, h), theta = rect_instance
        if w < h:
            temp = w
            w = h
            h = temp
            theta = theta + 90
        bbx_instance = np.array([x, y, w+1, h+1, theta]).astype(int)
        # bbx_instance = torch.tensor(bbx_instance)
        # bbx_region = cv2.imread(bbx_region_path, cv2.IMREAD_GRAYSCALE)
        # _, bbx_region_mask = cv2.threshold(bbx_region, 128, 255, cv2.THRESH_BINARY)

        dilated_shadow_mask = cv2.resize(shadow_mask, (width_mask, height_mask))
        kernel = np.ones((6,6), np.uint8)
        dilated_shadow_mask = cv2.dilate(dilated_shadow_mask, kernel, iterations=1)
        
        shadowfree_img = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)

        cls_input = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        source = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)

        # Normalize source images to [0, 1].
        cls_input = cls_input.astype(np.float32) / 255.0
        source = source.astype(np.float32) / 255.0
        shadow_mask = shadow_mask.astype(np.float32) / 255.0
        dilated_shadow_mask = dilated_shadow_mask.astype(np.float32) / 255.0
        object_mask = object_mask.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        mask_embeddings = torch.zeros((64, 2048), dtype=torch.float32)
        bbx_region = torch.zeros((512, 512), dtype=torch.float32)
        return dict(jpg=target, fg=bbx_instance, bbx=bbx_region, embeddings=mask_embeddings, txt=prompt, cls=cls_input, hint=source, shadowmask=shadow_mask, objectmask=object_mask, dilated_shadow_mask=dilated_shadow_mask)
    
class TestDataset(Dataset):
    def __init__(self, data_file_path, K=64, device = None):    
        self.data_root = data_file_path
        with open(os.path.join(self.data_root, 'test.txt'), 'r') as file:
            self.data = [line.rstrip('\n') for line in file] 
        self.k = K

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if '.' in self.data[idx]:
            pic_name = self.data[idx]
        else:
            pic_name = self.data[idx] + '.jpg'
        shadowfree_img_path = os.path.join(self.data_root, 'shadowfree_imgs', pic_name)
        object_mask_path = os.path.join(self.data_root, 'object_masks', pic_name)
        shadow_mask_path = os.path.join(self.data_root, 'shadow_masks', pic_name)
        shadow_img_path = os.path.join(self.data_root, 'shadow_imgs', pic_name)
        prompt = ''
        
        width, height = 512, 512
        
        shadowfree_img = cv2.imread(shadowfree_img_path)
        shadowfree_img = cv2.resize(shadowfree_img, (width, height))
        object_mask = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask = cv2.resize(object_mask, (width, height))
        shadow_mask = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        shadow_mask = cv2.resize(shadow_mask, (width, height))

        _, fg_instance_thresh = cv2.threshold(object_mask, 128, 255, cv2.THRESH_BINARY)
        contours_instance, _ = cv2.findContours(fg_instance_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        merged_contour_points_instance = np.concatenate(contours_instance)
        rect_instance = cv2.minAreaRect(merged_contour_points_instance)
        (x, y), (w, h), theta = rect_instance
        if w < h:
            temp = w
            w = h
            h = temp
            theta = theta + 90
        bbx_instance = np.array([x, y, w+1, h+1, theta]).astype(int)
        bbx_instance = torch.tensor(bbx_instance)

        shadowfree_img = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)
        target = shadowfree_img
        zt = target

        source = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        cls_input = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        cls_input = cls_input.astype(np.float32) / 255.0

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        shadow_mask = shadow_mask.astype(np.float32) / 255.0
        object_mask = object_mask.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        gt_img = cv2.imread(shadow_img_path)
        gt_img = cv2.resize(gt_img, (256, 256))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = (gt_img.astype(np.float32) / 127.5) - 1.0
        
        mask_embeddings = torch.zeros((self.k, 2048), dtype=torch.float32)
        bbx_region = torch.zeros((512, 512), dtype=torch.float32)

        shadow_mask_ = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
        shadow_mask_ = cv2.resize(shadow_mask_, (256, 256))
        shadow_mask_ = shadow_mask_.astype(np.float32) / 255.0
        
        shadowfree_img_ = cv2.imread(shadowfree_img_path)
        shadowfree_img_ = cv2.resize(shadowfree_img_, (256, 256))
        shadowfree_img_ = cv2.cvtColor(shadowfree_img_, cv2.COLOR_BGR2RGB)
        shadowfree_img_ = (shadowfree_img_.astype(np.float32) / 127.5) - 1.0
        
        object_mask_ = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask_ = cv2.resize(object_mask_, (256, 256))
        object_mask_ = object_mask_.astype(np.float32) / 255.0

        return dict(zt=zt, jpg=target, cls=cls_input, fg=bbx_instance, bbx=bbx_region, embeddings=mask_embeddings, txt=prompt, hint=source, shadowmask=shadow_mask, objectmask=object_mask, \
                     gt=gt_img, shadow_mask_ = shadow_mask_, \
                        img_name=pic_name, shadowfree_img_=shadowfree_img_, object_mask_=object_mask_)

class TestDataset_single(Dataset):
    def __init__(self, shadowfree_img_path, object_mask_path):    
        self.shadowfree_img_path = shadowfree_img_path
        self.object_mask_path = object_mask_path

    def __len__(self):
        return len(self.shadowfree_img_path)

    def __getitem__(self, idx):
        
        shadowfree_img_path = self.shadowfree_img_path
        object_mask_path = self.object_mask_path
        pic_name = os.path.basename(shadowfree_img_path)
        prompt = ''
        width, height = 512, 512
        shadowfree_img = cv2.imread(shadowfree_img_path)
        shadowfree_img = cv2.resize(shadowfree_img, (width, height))
        object_mask = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask = cv2.resize(object_mask, (width, height))
        _, fg_instance_thresh = cv2.threshold(object_mask, 128, 255, cv2.THRESH_BINARY)
        contours_instance, _ = cv2.findContours(fg_instance_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        merged_contour_points_instance = np.concatenate(contours_instance)
        rect_instance = cv2.minAreaRect(merged_contour_points_instance)
        (x, y), (w, h), theta = rect_instance
        if w < h:
            temp = w
            w = h
            h = temp
            theta = theta + 90
        bbx_instance = np.array([x, y, w+1, h+1, theta]).astype(int)
        bbx_instance = torch.tensor(bbx_instance)
        shadowfree_img = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)
        source = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        cls_input = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        cls_input = cls_input.astype(np.float32) / 255.0
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        mask_embeddings = torch.zeros((64, 2048), dtype=torch.float32)
        bbx_region = torch.zeros((512, 512), dtype=torch.float32)

        return dict(jpg=target, cls=cls_input, fg=bbx_instance, bbx=bbx_region, embeddings=mask_embeddings, img_name=pic_name, txt=prompt, hint=source)
