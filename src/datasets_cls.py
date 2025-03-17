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
        size = 256
        # load image
        composite_img = imread(self.composite_data[index])
        composite_img = cv2.resize(composite_img, (size, size))
        label = self.get_label_for_image(self.composite_data[index], self.label_txt)

        fg_instance = cv2.imread(self.fg_instance_data[index], cv2.IMREAD_GRAYSCALE)
        fg_instance = cv2.resize(fg_instance, (size, size))
        _, fg_instance_thresh = cv2.threshold(fg_instance, 128, 255, cv2.THRESH_BINARY)

        return self.to_tensor(composite_img), self.to_tensor(fg_instance_thresh), label

    def get_label_for_image(self, image_path, label_file):
        image_name = os.path.basename(image_path)
        identifier = os.path.splitext(image_name)[0]

        with open(label_file, 'r') as f:
            lines = f.readlines()
    
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                if parts[0] == identifier:
                    return torch.tensor(int(parts[1]), dtype=torch.long)  
    
        print(f"Error: Image {image_path}")
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

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=False
            )

            for item in sample_loader:
                yield item