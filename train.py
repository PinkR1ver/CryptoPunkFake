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

basePath = r'C:\Users\83549\Github Projects\CryptoPunkFake\imgs'
weightPath = r'C:\Users\83549\Github Projects\CryptoPunkFake\params'
batchSize = 64
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
    cryptoPunkDataSet = ImageDataSet(basePath)
    cryptoPunkDataLoader = DataLoader(cryptoPunkDataSet, batch_size=batchSize, shuffle=True)


    generatorNet = Generator().to(device)
    discriminatorNet = Discriminator().to(device)

    if os.path.exists(os.path.join(weightPath, 'Generator.pth')):
        generatorNet.load_state_dict(torch.load(os.path.join(weightPath, 'Generator.pth')))
        print("Generator:Loading Weight Success")
    else:
        print("Generator:Loading Weight Failed")

    
    if os.path.exists(os.path.join(weightPath, 'Discriminator.pth')):
        discriminatorNet.load_state_dict(torch.load(os.path.join(weightPath, 'Discriminator.pth')))
        print("Discriminator:Loading Weight Success")
    else:
        print("Discriminator:Loading Weight Failed")

    # generatorNet.apply(WeightsInit)
    
    # discriminatorNet.apply(WeightsInit)

    
    criterion = nn.BCELoss()
    fixedNoise = torch.randn(64, nz, 1, 1, device=device)

    readLabel = 1
    fakeLabel = 0

    optimizerD = optim.Adam(discriminatorNet.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generatorNet.parameters(), lr=lr, betas=(beta1, 0.999))

    epoch = 1
    while True:

        for i, cryptoPunkImage in enumerate(cryptoPunkDataLoader):
            
            ### (1) Update Disciminator Net: Maximize log(D(x)) + log(1-D(G(z)))

            ## Train with all-real batch
            discriminatorNet.zero_grad()

            realImage = cryptoPunkImage.to(device)
            # realImage = torch.unsqueeze(realImage, 0)

            BatchSize = realImage.size(0)
            label = torch.full((BatchSize,),readLabel, dtype=torch.float, device=device)
            output = discriminatorNet(realImage).view(-1)
            #print(f'{output.shape} & {label.shape}')
            #size := batchSize
            errDReal = criterion(output, label)
            errDReal.backward()
            Dx = output.mean().item()


            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(BatchSize, nz, 1, 1, device=device)
            fakeImage = generatorNet(noise)

            #print(fakeImage.shape) fakeImage.shape:[BatchSize, channel, image_height, image_weight]
            label.fill_(fakeLabel)
            output = discriminatorNet(fakeImage.detach()).view(-1)
            errDFake = criterion(output, label)
            errDFake.backward()
            DG_z1 = output.mean().item()
            errD = errDReal + errDFake
            optimizerD.step()

            ### (2) Update G network maximize log(D(G(z)))
            
            generatorNet.zero_grad()
            label.fill_(readLabel)
            output = discriminatorNet(fakeImage).view(-1)
            errG = criterion(output, label)
            errG.backward()
            DG_z2 = output.mean().item()
            optimizerG.step()

            if i % 5 == 0:
                print(f'epoch:{epoch},prograss:{i}\{len(cryptoPunkDataLoader)}\tLoss_D:{errD.item()}\tLoss_G:{errG.item()}')
                print(f'D(x):{Dx}\tD(G(z)):{DG_z1}/{DG_z2}')
            
            if i% 50 == 0:
                torch.save(generatorNet.state_dict(), os.path.join(weightPath, 'Generator.pth'))
                torch.save(discriminatorNet.state_dict(), os.path.join(weightPath, 'Discriminator.pth'))


        epoch += 1

            
            

