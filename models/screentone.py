import torch
from torch.utils.data import Dataset
import random
import os
from .selectRect import SelectAvailableRange
# import cv2
from PIL import Image
from torchvision import transforms

class ScreenTone(Dataset):
    
    def __init__(self, root, split, transform, img_size=32, select_rect=False):
        self.root = root
        self.img_size = img_size
        self.files = []
        self.split = split
        for root, _, files in os.walk(root):
            for file in files:
                if file[-4:] in [".jpg", ".png"]:
                    self.files.append(os.path.join(root, file))
        if split == 'train':
            self.files = [self.files[i] for i in range(len(self.files))if i % 10 < 8]
        elif split == 'test':
            self.files = [self.files[i] for i in range(len(self.files)) if i % 10 >= 8]
        elif split == 'predict':
            self.select_rect = select_rect
            if select_rect:
                self.selecter = SelectAvailableRange(img_size)
                files = []
                for file in self.files:
                    if self.selecter.check(file):
                        files.append(file)
                self.files = files
        else:
            raise ValueError('Undefined dataset split')
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # image = Image.open(self.files[idx]).convert('RGB')
        if self.split == 'predict':
            image = Image.open(self.files[idx]).convert('L')
            if self.select_rect:
                rw, rh = self.selecter.getRandomXY(self.files[idx])
                half_size = self.img_size // 2
                image = image.crop((rw - half_size, rh - half_size, rw + half_size, rh + half_size))
            image = self.transform(image.split()[0])
            return image, self.files[idx]
        image = Image.open(self.files[idx]).convert('L')
        double_image = transforms.RandomCrop(self.img_size * 2)(image)
        image = transforms.RandomCrop(self.img_size)(double_image)
        return transforms.ToTensor()(double_image), transforms.ToTensor()(image)