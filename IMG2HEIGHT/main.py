import argparse
import os
import copy
import torchvision
import torch
import datetime
from tqdm import trange
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from utils import (Nptranspose,Rotation,H_Mirror,V_Mirror)
#from datasets import TrainDataset
from datasets_chl import TrainDataset
import trainer
from network.Net import Resnet50




def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    src_start = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_file',  default="../IEEE_data/dataset/train/image/", type=str)
    parser.add_argument('--train_label_file',  default="../IEEE_data/dataset/train/dsm/", type=str)
    parser.add_argument('--test_image_file', default="../IEEE_data/dataset/test/image/", type=str)
    parser.add_argument('--test_label_file', default="../IEEE_data/dataset/test/dsm/", type=str)

    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    '''
    设置模型、损失函数和优化器
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Resnet50()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr= args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.9)
    
    '''
    Prepare data
    '''
    train_composed = torchvision.transforms.Compose([Rotation(),H_Mirror(),V_Mirror(),Nptranspose()])
    train_dataset = TrainDataset(args.train_image_file,args.train_label_file,train_composed)
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,drop_last=True)

    test_composed = torchvision.transforms.Compose([Nptranspose()])
    test_dataset = TrainDataset(args.test_image_file,args.test_label_file,test_composed)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=6,num_workers=args.num_workers,
                                  pin_memory=True,drop_last=True)
    '''
    Training model
    '''

    print("len(train_dataloader)",len(train_dataloader))
    print("len(eval_dataloader)",len(test_dataloader))
    if True:
        print("begin training!")
        trainer.train_model(model, args, train_dataloader, test_dataloader,
            criterion,optimizer,device)
    src_end = datetime.datetime.now()
    print("run time {}".format(src_end-src_start))