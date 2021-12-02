import torch
from PIL import Image
from net import *
import torchvision
import os

if torch.cuda.is_available():
    device = 'cuda'
    print('Using cuda')
else:
    device = 'cpu'
    print('Using CPU')

weightPath = '/home/pinkr1ver/Documents/Github Projects/GAN/params'
savePath = '/home/pinkr1ver/Documents/Github Projects/GAN/results'

nz = 100

if __name__ == '__main__':

    generatorNet = Generator().to(device)
    generatorNet.load_state_dict(torch.load(os.path.join(weightPath, 'Generator.pth')))

    nImages = 100

    for i in range(1, nImages):
        noise = torch.randn(1, nz, 1, 1, device=device)
        #print(noise)
        Results = generatorNet(noise)
        
        torchvision.utils.save_image(Results, f'{savePath}/{i}.png')


    