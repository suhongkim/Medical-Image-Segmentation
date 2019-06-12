import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import torchvision.transforms.functional as TF


class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

        self.unet_in_size = 572
        self.unet_out_size = 388
        
    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            start = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            start = n_train
            endId = len(self.data_files)

        while current < endId:
            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            # Load image
            data_image = Image.open(self.data_files[current])
            label_image = Image.open(self.label_files[current])
            
            # Resize image from 1024X1024 to output size(388X388)
            data_image = data_image.resize((self.unet_out_size, self.unet_out_size))            
            label_image = label_image.resize((self.unet_out_size, self.unet_out_size))
            
            # Data Augmentation
            if self.mode == 'train':
                data_image, label_image = self.applyDataAugmentation(data_image, label_image)

            # Change images to numpy array
            data_image = np.divide(np.asarray(data_image, dtype=np.float32), 255.)
            label_image = np.asarray(label_image, dtype=np.float32)

            # Overlap Tile strategy - Mirroring
            # tile data_image into 3X3 grid
            tiled1 = np.concatenate((np.flipud(np.fliplr(data_image)), 
                                     np.flipud(data_image), 
                                     np.flipud(np.fliplr(data_image))), axis=1)
            tiled2 = np.concatenate((np.fliplr(data_image), 
                                     data_image, 
                                     np.fliplr(data_image)), axis=1)
            tiled3 = np.concatenate((np.flipud(np.fliplr(data_image)), 
                                     np.flipud(data_image), 
                                     np.flipud(np.fliplr(data_image))), axis=1)
            tiled_image = np.concatenate((tiled1, tiled2, tiled3), axis=0)
            diff = (3*self.unet_out_size - self.unet_in_size)//2
            data_image = tiled_image[diff:tiled_image.shape[0]-diff, diff:tiled_image.shape[1]-diff]

            current += 1
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def applyDataAugmentation(self, img, target):
        # Flip & Rotate Image
        random_flip = random.randint(-1, 5)  # -1 is None
        img = img.transpose(random_flip) if random_flip > -1 else img
        target = target.transpose(random_flip) if random_flip > -1 else target

        # Zoom images
        random_crop = random.randint(5, 10)
        width, height = img.size
        img = ImageOps.crop(img, img.size[1] // random_crop).resize((width, height))
        target = ImageOps.crop(target, target.size[1] // random_crop).resize((width, height))
        
        # Gamma Correction
        random_gamma = random.uniform(0.5, 1.5)
        img = TF.adjust_gamma(img,random_gamma, gain=1)
       
        return img, target

def plotImages(img, img_aug):
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_aug)
    plt.show()

if __name__ == "__main__":     
    pass
#     img = Image.open("/home/suhongk/Documents/VC_Assignment1_2/data/cells/scans/BMMC_4.tif")
#     img = img.resize((388, 388))


