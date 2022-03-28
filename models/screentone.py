from torch.utils.data import Dataset
from PIL import Image
import sys
from .SCGenerator import genArg, genImg

class ScreenTone(Dataset):
    
    def __init__(self, root, split, transform, img_size=32, select_rect=False):
        self.root = root
        self.img_size = img_size
        self.split = split
        self.transform = transform

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        arg = genArg(self.img_size, self.img_size)
        image = Image.fromarray(genImg(arg, self.img_size, self.img_size) / 255)
        # image = Image.open(self.files[idx]).convert('L')
        image = self.transform(image)
        return image, []
        # return transforms.ToTensor()(double_image), transforms.ToTensor()(image)