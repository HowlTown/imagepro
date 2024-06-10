import os
import numpy as np
from skimage import io

import torchvision
from torch.utils.data import Dataset,DataLoader

from utils import (Nptranspose,Rotation,H_Mirror,V_Mirror)
# from utils import (RandomCrop,StdCrop)

class TrainDataset(Dataset):
    def __init__(self,image_dir,label_dir,transform=None):

        self.label_dir = label_dir
        self.image_dir = image_dir 
        
        self.data = []
        self.transform = transform

        files = os.listdir(self.label_dir)
        for item in files:
            if item.endswith(".tif"):
                self.data.append(item.split(".tif")[0][3:])
        self.data.sort()

                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
 
        image = self.image_dir + "image" + self.data[index] + ".tif"
        label = self.label_dir + "dsm" + self.data[index]+".tif"

        image = io.imread(image)
        # print("image",image.shape)

        # image = np.reshape(image,(image.shape[0],image.shape[1],1))
        label = io.imread(label) 
        label = np.reshape(label,(label.shape[0],label.shape[1],1))

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        # print(image)
    
        image = image/255.0
        # image = image.clip(min=0,max=1)
        label=label

        sample = {}
        sample["image"] = image
        sample["label"] = label
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


