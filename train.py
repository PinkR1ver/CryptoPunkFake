import os
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import torchvision
from data import *
from net import *
import numpy as np

if torch.cuda.is_available():
    device = 'cuda'
    print('Using cuda')
else:
    device = 'cpu'
    print('Using CPU')

basepath = f'/home/pinkr1ver/Documents/Github Projects/GAN/imgs'
batchSize = 8
imageSize = 64

#Number of Channels
nc = 4

#Size of z latent vector, aka size of generator input
nz = 100

# numEpochs = 5

#Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for optimizers
beta1 = 0.5

if __name__ == '__main__':
    cryptoPunkDataSet = ImageDataSet(basepath)
    cryptoPunkDataLoader = DataLoader(cryptoPunkDataSet, batch_size=batchSize, shuffle=True)


    generatorNet = Generator().to(device)
    discriminatorNet = Discriminator().to(device)

    generatorNet.apply(WeightsInit)
    discriminatorNet.apply(WeightsInit)

    
    criterion = nn.BCELoss()
    fixedNoise = torch.randn(64, nz, 1, 1, device=device)

    readLabel = 1
    fakeLabel = 0

    optimizerD = optim.Adam(discriminatorNet.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generatorNet.parameters(), lr=lr, betas=(beta1, 0.999))

    epoch = 1
    while True:

        for i, cryptoPunkImage in enumerate(cryptoPunkDataLoader):
            
            # (1) Update Disciminator Net: Maximize log(D(x)) + log(1-D(G(z)))

            ## Train with all-real batch
            discriminatorNet.zero_grad()

            realImage = cryptoPunkImage[0].to(device)
            realImage = torch.unsqueeze(realImage, 0)

            BSize = realImage.size(0)
            label = torch.full((BSize,),readLabel, dtype=torch.float, device=device)
            output = discriminatorNet(realImage)
            print(output.size())
            
            

