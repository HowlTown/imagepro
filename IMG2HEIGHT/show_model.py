import numpy as np
import sys
import torch, torchvision  
from skimage import io
from torch import nn
from torch.utils.data import DataLoader  
from torchvision.transforms import Compose, ToTensor, Normalize
sys.path.insert(0,'.')
import network
from datasets_chl import TrainDataset
from utils import (Nptranspose,Rotation,H_Mirror,V_Mirror)
import cal_acc
from sklearn.metrics import mean_squared_error  

def show_result(num_channels,model_path,image_dir,label_dir):
    model = None
    with open(model_path,'rb') as f:
        model = torch.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  
    model.eval()  # 设置模型为评估模式 
    # 准备数据  
    test_composed = None
    test_dataset = TrainDataset(num_channels,image_dir,label_dir,test_composed)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=1,num_workers=1,
                                  pin_memory=True,drop_last=True) 
    result = []
    with torch.no_grad():  # 不需要计算梯度  
        for index, sample in enumerate(test_dataloader): 
            image,label = sample['image'],sample['label']
            image = image.to(torch.float32)  # torch.float32
            label = label.to(torch.float32)
            image = image.to(device)
            label = label.to(device) 
            
            # 假设你的模型输出也是与targets相同形状的tensor  
            output,_,_,_ = model(image)
            result.append(output.cpu().numpy().astype('float32') * 255.0)
    return result
    

def show_result(model_3_path, model_4_path, image_dir, label_dir):
    #加载模型
    model_3,model_4 = None,None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(model_3_path,'rb') as f3:
        model_3 = torch.load(f3)
    model_3 = model_3.to(device)
    model_3.eval()  # 设置模型为评估模式
    
    
    with open(model_4_path,'rb') as f4:
        model_4 = torch.load(f4)
    model_4 = model_4.to(device)  
    model_4.eval()  # 设置模型为评估模式 
    
    # 准备数据  
    test_composed = None
    test_3_dataset = TrainDataset(3,image_dir,label_dir,test_composed)
    test_4_dataset = TrainDataset(4,image_dir,label_dir,test_composed)
    test_3_dataloader = DataLoader(dataset = test_3_dataset, batch_size=1,num_workers=1,
                                  pin_memory=True,drop_last=True)
    test_4_dataloader = DataLoader(dataset = test_4_dataset, batch_size=1,num_workers=1,
                                  pin_memory=True,drop_last=True)

    rgb, dsm, depth, output_3, output_4 = [], [], [], [], []
    with torch.no_grad():  # 不需要计算梯度  
        for index, sample in enumerate(test_3_dataloader): 
            image,label = sample['image'],sample['label']
            rgb.append(image * 255.0)
            dsm.append(label * 255.0)
            image = image.to(torch.float32)  # torch.float32
            image = image.to(device)
            print(image.shape)
            output,_,_,_ = model_3(image)  
            output_3.append(output.cpu().numpy().astype('float32') * 255.0)
    with torch.no_grad():  # 不需要计算梯度  
        for index, sample in enumerate(test_4_dataloader): 
            image = sample['image']
            depth.append(image[:, :, 3] * 255.0)
            image = sample.to(torch.float32)  # torch.float32
            image = image.to(device)
            
            output,_,_,_ = model_4(image)  
            output_4.append(output.cpu().numpy().astype('float32') * 255.0)
    
    return rgb,dsm,depth,output_3,output_4

if __name__ == "__main__":
    model_3 = "model/seg=49_3.pkl"
    model_4 = "model/seg49.pkl"
    image = "../IEEE_data/dataset_show/image/"
    label = "../IEEE_data/dataset_show/dsm/"
    rgb,dsm,depth,output_3,output_4 = show_result(model_3, model_4, image, label)