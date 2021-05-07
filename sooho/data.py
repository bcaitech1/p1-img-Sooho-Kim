import os
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2

class MaskDataset(Dataset):
    def __init__(self, data_root, input_size=224, transform=None, phase='train'):
        super(MaskDataset, self).__init__()
        self.input_size = input_size
        self.transform = transform
        if phase == 'train':
            file_list = self.augment_dataset(data_root)
            self.file_list = file_list
        else:
            file_list = glob(data_root + '/*' + '/*')
            self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_name = os.path.basename(self.file_list[index])
        img_path = self.file_list[index]

        # PIL
        image = Image.open(img_path)
#         # cv2
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = image/255.
        
        label = self._get_label(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)
    
    def _get_label(self, img_path):
        """
        image에 대한 라벨을 구하는 함수입니다.
        """
        target = img_path.split('/')[-2].split('_')
        if (target[1]=='male')&(int(target[3])<30):
            folder_label = 0
        elif (target[1]=='male')&(int(target[3]) >= 30)&(int(target[3]) < 58):
            folder_label = 1
        elif (target[1]=='male')&(int(target[3]) >= 58):
            folder_label = 2
        elif (target[1]=='female')&(int(target[3]) < 30):
            folder_label = 3
        elif (target[1]=='female')&(int(target[3]) >= 30)&(int(target[3]) < 58):
            folder_label = 4
        elif (target[1]=='female')&(int(target[3]) >= 58):
            folder_label = 5
        if 'incorrect' in os.path.basename(img_path):
            label = folder_label + 6
        elif 'normal' in os.path.basename(img_path):
            label = folder_label + 12
        else:
            label = folder_label
        return label
    
    def set_transform(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        self.transform = transform