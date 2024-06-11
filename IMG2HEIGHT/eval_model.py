import numpy as np
import sys
import torch, torchvision  
from torch import nn
from torch.utils.data import DataLoader  
from torchvision.transforms import Compose, ToTensor, Normalize
sys.path.insert(0,'.')
import network
from datasets_chl import TrainDataset
from utils import (Nptranspose,Rotation,H_Mirror,V_Mirror)
import cal_acc

from sklearn.metrics import mean_squared_error  

def cal_rmse(img1, img2):
    img1_np = img1.cpu().numpy().astype('float32')  
    img2_np = img2.cpu().numpy().astype('float32')  
      
    # 计算整个批次的MSE，然后取其平方根得到RMSE  
    mse = mean_squared_error(img1_np.reshape(-1), img2_np.reshape(-1))  
    rmse = np.sqrt(mse)  
    return rmse

def eval_model(model_path, image_dir, label_dir):
    model = None
    with open(model_path,'rb') as f:
        model = torch.load(f)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  
    model.eval()  # 设置模型为评估模式  
    
    # 准备数据  
    test_composed = torchvision.transforms.Compose([Nptranspose()])
    test_dataset = TrainDataset(3,image_dir,label_dir,test_composed)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=2,num_workers=1,
                                  pin_memory=True,drop_last=True)

    
    # 评估模型  
    criterion = nn.MSELoss()  # 假设你使用均方误差作为损失函数  
    total_loss = 0.0 
    total_rmse = 0.0 
    total_rmse1 = 0.0
    
    
    with torch.no_grad():  # 不需要计算梯度  
        for index, sample in enumerate(test_dataloader): 
            image,label = sample['image'],sample['label']
            image = image.to(torch.float32)  # torch.float32
            label = label.to(torch.float32)
            image = image.to(device)
            label = label.to(device) 
            
            # 假设你的模型输出也是与targets相同形状的tensor  
            output,_,_,_ = model(image)  
            
            loss = criterion(output, label)   
            rmse = cal_rmse(output,label)
            rmse1 = cal_acc.cal_rmse(output,label)
            
            total_loss += loss.item() 
            total_rmse += rmse
            total_rmse1 += rmse1
            
    
    # 计算平均损失  
    num_data = len(test_dataloader.dataset)  
    avg_loss = total_loss / num_data
    avg_rmse = total_rmse / num_data
    avg_rmse1 = total_rmse1 / num_data
    print('Average MSE loss on the test set: {:.4f}'.format(avg_loss))
    print(avg_rmse,avg_rmse1)

if __name__ == "__main__":
    model = "model\seg=49_3.pkl"
    image = "../IEEE_data/dataset/test/image/"
    label = "../IEEE_data/dataset/test/dsm/"
    eval_model(model,image,label)