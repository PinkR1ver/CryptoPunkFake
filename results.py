import torch
from PIL import Image
import PIL
from torchvision.utils import save_image
from net import *
import torchvision
from torchvision import transforms
import os
import numpy as np


if torch.cuda.is_available():
    device = 'cuda'
    print('Using cuda')
else:
    device = 'cpu'
    print('Using CPU')

weightPath = r'C:\Users\83549\Github Projects\CryptoPunkFake\params'
savePath = r'C:\Users\83549\Github Projects\CryptoPunkFake\results'

nz = 100

imageSize = 1024

transform = transforms.Compose([
    transforms.Resize((imageSize, imageSize), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
])

if __name__ == '__main__':

    generatorNet = Generator().to(device)
    generatorNet.load_state_dict(torch.load(os.path.join(weightPath, 'Generator.pth')))

    nImages = 500

    for i in range(1, nImages):
        noise = torch.randn(1, nz, 1, 1, device=device)
        #print(noise)
        Results = generatorNet(noise)

        saveImage = transform(Results)
        
        torchvision.utils.save_image(saveImage, f'{savePath}\\{i}.png')


    