import os
import torch
import torch.nn as nn
import logging
import datetime
import numpy as np
from tqdm import trange
#from skimage.measure import compare_ssim
from torchvision.transforms import Resize


from cal_acc import cal_psnr,cal_ssim,cal_mae,cal_rmse,cal_zncc
logging.basicConfig(level=logging.DEBUG,
                        filename='./my.log',
                        filemode='a',
                    format='%(asctime)s %(filename)s[line:%(lineno)d ] %(levelname)s %(message)s', #时间 文件名 line:行号  levelname logn内容
                    datefmt='%d %b %Y,%a %H:%M:%S', #日 月 年 ，星期 时 分 秒
                         )

def train_model(model,args,train_dataloader,test_dataloader,
    criterion,optimizer,device):
    #torch_resize2 = Resize([128, 128])
    #torch_resize3 = Resize([64, 64])
    #torch_resize4 = Resize([32, 32])
    torch_resize2 = Resize([256, 256])
    torch_resize3 = Resize([128, 128])
    torch_resize4 = Resize([64, 64])
    for epo in trange(args.num_epochs):
        epoch_start = datetime.datetime.now()

        train_loss = 0.0
        model.train()
        model=model.to(device)
        for index, sample in enumerate(train_dataloader):
            image,label = sample['image'],sample['label']
            image = image.to(torch.float32) 
            # label = sample["label"]
            label = label.to(torch.float32)

            image = image.to(device)
            label = label.to(device)

            label2 = torch_resize2(label)
            label3 = torch_resize3(label)
            label4 = torch_resize4(label)

            optimizer.zero_grad()
            output,x4, x3, x2= model(image)
            loss1 = criterion(output, label)
            loss2 = criterion(x2, label2)

            loss3 = criterion(x3, label3)

            loss = loss1 + loss2 + loss3
            # print(loss.item())

            # loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            iter_loss = loss.item()
            train_loss += iter_loss


        
            if np.mod(index+1, 500) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
        '''
        Validation model
        '''
        model.eval()
        eval_loss = 0.0
        eval_psnr = 0.0
        eval_ssim = 0.0
        eval_mae=0.0
        eval_rmse=0.0
        eval_zncc = 0.0

        with torch.no_grad():
            for index, sample in enumerate(test_dataloader):
                image,label = sample['image'],sample['label']
                image = image.to(torch.float32)  # torch.float32
                # label = sample["label"]
                label = label.to(torch.float32)

                image = image.to(device)
                label = label.to(device)


                # output,_,_,_ = model(image)
                output,_,_,_= model(image)

                loss = criterion(output, label)
                iter_loss = loss.item()
                # all_test_iter_loss.append(iter_loss)
                eval_loss += iter_loss

                print(output.shape,label.shape)
                psnr=cal_psnr(output, label)
                ssim=cal_ssim(output, label)
                mae=cal_mae(output, label)
                rmse=cal_rmse(output, label)
                zncc=cal_zncc(output,label)

                eval_psnr+=psnr.item()
                eval_ssim+=ssim
                eval_mae+=mae
                eval_rmse+=rmse
                eval_zncc+=zncc

        epoch_end = datetime.datetime.now()
        logging.info("End of one epoch,Running time={},train_loss={},eval_loss={},eval_psnr={},eval_ssim={}".format(epoch_end-epoch_start,
            round(train_loss/len(train_dataloader),6),round(eval_loss/len(test_dataloader),6),round(eval_psnr/len(test_dataloader),4),round(eval_ssim/len(test_dataloader),4)))

        print("End of one epoch,Running time={},train_loss={},eval_loss={},eval_psnr={},eval_ssim={},eval_mae={},eval_rmse={},eval_zncc={}".format(epoch_end-epoch_start,
            round(train_loss/len(train_dataloader),6),round(eval_loss/len(test_dataloader),6),round(eval_psnr/len(test_dataloader),4),
            round(eval_ssim/len(test_dataloader),4),round(eval_mae/len(test_dataloader),4),round(eval_rmse/len(test_dataloader),4),round(eval_zncc/len(test_dataloader),4)))

        torch.save(model,
                   "./model/seg={}_train_loss={}_eval_loss={}_eval_psnr={}_eval_ssim={}_eval_mae={}_eval_rmse={}_eval_zncc={}.pkl".
            format(epo,round(train_loss/len(train_dataloader),6),round(eval_loss / len(test_dataloader),6),
            round(eval_psnr/len(test_dataloader),4),round(eval_ssim/len(test_dataloader),4),round(eval_mae/len(test_dataloader),4)
            ,round(eval_rmse/len(test_dataloader),4),round(eval_zncc/len(test_dataloader),4)))
