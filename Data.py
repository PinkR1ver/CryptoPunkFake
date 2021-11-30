import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

transform = transforms.Compose([
    transforms.ToTensor()
])

ImageSize = 64

class ImageDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(path)

    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        imageName = self.name[index]
        image = getImage(os.path.join(self.path, imageName), ImageSize)
        return transform(image)

if __name__ == '__main__':
    CryptoPunkDataSet = ImageDataSet(f'/home/pinkr1ver/Documents/Github Projects/GAN/imgs')
    print(CryptoPunkDataSet)
        
