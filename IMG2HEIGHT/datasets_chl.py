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
        # 这里在记录文件名称
        for item in files:
            if item.endswith(".tif"):
                self.data.append(item.split(".tif")[0])
        self.data.sort()

                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        # 数据文件路径
        image_path = self.image_dir + self.data[index] + ".npy"
        label_path = self.label_dir + self.data[index] + ".tif"
        
        # 读取image数据并转置成512*512*4
        image = np.load(image_path)
        image = np.transpose(image,(1,2,0))
        
        # 读取label数据并转置成512*512*1
        label = io.imread(label_path) 
        label = np.reshape(label,(label.shape[0],label.shape[1],1))
        
        # 格式化为float
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        # 数据归一化
        image = image / 255.0
        label = label / 255.0
        
        # 样本
        sample = {}
        sample["image"] = image
        sample["label"] = label
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


