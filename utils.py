from PIL import Image

def getImage(path,imageSize):
    image = Image.open(path)
    image = image.resize((imageSize, imageSize), resample=Image.NEAREST)
    return image