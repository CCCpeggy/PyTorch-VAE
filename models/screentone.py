from torch.utils.data import Dataset
from PIL import Image
import sys
from .NoiseGenerator import *

class ScreenTone(Dataset):
    
    def __init__(self, root, split, transform, img_size=32, select_rect=False):
        self.root = root
        self.img_size = img_size
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arg = genArg()
        img = genImg(arg)
        cv2.imwrite("output.png", img)
        image = Image.open(self.files[idx]).convert('L')
        image = self.transform(image)
        return image, []
        # return transforms.ToTensor()(double_image), transforms.ToTensor()(image)