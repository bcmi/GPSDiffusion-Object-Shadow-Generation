import json
import os
import random
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
from PIL import Image
from imageio import imread
from skimage.color import rgb2gray, gray2rgb
import sys
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, rank, config, name_list, composite_flist, fg_instance_flist, fg_shadow_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.rank = rank
        self.training = training
        self.composite_data = self.load_flist(name_list, composite_flist)
        self.fg_instance_data = self.load_flist(name_list, fg_instance_flist)
        self.fg_shadow_data = self.load_flist(name_list, fg_shadow_flist)
        self.input_size = config.INPUT_SIZE
        self.label_txt = config.label_txt

        if self.rank == 0:
            print('training:{} data_list:{}'.format(training, composite_flist))
            
    def __len__(self):
        return len(self.composite_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            if self.rank == 0:
                print('loading error: ' + self.data[index])
                item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.composite_data[index]
        return os.path.basename(name)

    def load_item(self, index):

        composite_img = imread(self.composite_data[index])
        image_name = os.path.basename(self.composite_data[index])
        identifier = os.path.splitext(image_name)[0]

        fg_instance = cv2.imread(self.fg_instance_data[index], cv2.IMREAD_GRAYSCALE)
        _, fg_instance_thresh = cv2.threshold(fg_instance, 128, 255, cv2.THRESH_BINARY)

        fg_shadow = cv2.imread(self.fg_shadow_data[index], cv2.IMREAD_GRAYSCALE)
        _, fg_shadow_thresh = cv2.threshold(fg_shadow, 128, 255, cv2.THRESH_BINARY)

        contours_instance, _ = cv2.findContours(fg_instance_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours_shadow, _ = cv2.findContours(fg_shadow_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        merged_contour_points_instance = np.concatenate(contours_instance)
        merged_contour_points_shadow = np.concatenate(contours_shadow)

        rect_instance = cv2.minAreaRect(merged_contour_points_instance)
        (x, y), (w, h), theta = rect_instance
        if w < h:
            temp = w
            w = h
            h = temp
            theta = theta + 90
        bbx_instance = np.array([x, y, w+1, h+1, theta]).astype(int)
        
        rect_shadow = cv2.minAreaRect(merged_contour_points_shadow)
        (x, y), (w, h), theta = rect_shadow
        if w < h:
            temp = w
            w = h
            h = temp
            theta = theta + 90
        bbx_shadow = np.array([x, y, w+1, h+1, theta]).astype(int)

        t = torch.zeros((5))
        t[0] = (bbx_shadow[0] - bbx_instance[0]) / bbx_instance[2]
        t[1] = (bbx_shadow[1] - bbx_instance[1]) / bbx_instance[3]
        t[2] = np.log(bbx_shadow[2] / bbx_instance[2])
        t[3] = np.log(bbx_shadow[3] / bbx_instance[3])
        t[4] = (bbx_shadow[4] - bbx_instance[4]) * np.pi / 180

        return self.to_tensor(composite_img), self.to_tensor(fg_instance_thresh), \
            torch.tensor(bbx_instance), torch.tensor(bbx_shadow), t, identifier

    def get_label_for_image(self, image_path, label_file):
        image_name = os.path.basename(image_path)
        identifier = os.path.splitext(image_name)[0] 

        with open(label_file, 'r') as f:
            lines = f.readlines()
    
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                if parts[0] == identifier:
                    return int(parts[1]) 
    
        print(f"Error: image {image_path}")
        sys.exit(1)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
    
    def load_flist(self, flist_path, image_folder):
        if flist_path is None:
            return []

        if isinstance(flist_path, str) and os.path.isfile(flist_path):
            with open(flist_path, 'r') as file:
                file_list = file.read().splitlines()

            image_paths = [os.path.join(image_folder, file_name) for file_name in file_list]
            return image_paths
        
    def load_flist1(self, flist):
        if flist is None:
            return []

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    # return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                    return np.genfromtxt(flist, dtype=np.str)
                except:
                    return [flist]

        with open(flist, 'r') as j:
            f_list = json.load(j)
            return f_list


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=False
            )

            for item in sample_loader:
                yield item
