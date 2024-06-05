import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
import logging
import datetime
import numpy as np
from tqdm import trange
from skimage.measure import compare_ssim
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
def cal_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def cal_ssim(img1,img2):
    ssim_list=[]
    for i in range(img1.shape[0]):
        x=img1[i][:].cpu().numpy().transpose(1,2,0).astype('float32')
        y=img2[i][:].cpu().numpy().transpose(1,2,0).astype('float32')
        ssim,diff=compare_ssim(x, y, full=True,multichannel=True)
        ssim_list.append(ssim)
    return np.mean(ssim_list)
def cal_mae(img1, img2):
	mae_list=[]
	for i in range(img1.shape[0]):
		x=img1[i][:].squeeze(0).cpu().numpy().astype('float32')
		y=img2[i][:].squeeze(0).cpu().numpy().astype('float32')
		# print(x.shape)
		mae=metrics.mean_absolute_error(x, y)
		mae_list.append(mae)
	return np.mean(mae_list)

def cal_rmse(img1, img2):
	rmse_list=[]
	for i in range(img1.shape[0]):
		x=img1[i][:].squeeze(0).cpu().numpy().astype('float32')
		y=img2[i][:].squeeze(0).cpu().numpy().astype('float32')
		# ase=mean_squared_error(x, y)
		rmse=np.sqrt(mean_squared_error(x, y))
		rmse_list.append(rmse)
	return np.mean(rmse_list)


def cal_mre(img1, img2):
	mre_list=[]
	for i in range(img1.shape[0]):
		x=img1[i][:].squeeze(0).cpu().numpy().astype('float32')
		y=img2[i][:].squeeze(0).cpu().numpy().astype('float32')
		# ase=mean_squared_error(x, y)
		mre=np.abs((y-x)/(y+1))
		mre_list.append(mre)
	return np.mean(mre_list)


def cal_zncc(img1, img2):
	zncc_list=[]
	for i in range(img1.shape[0]):
		x=img1[i][:].squeeze(0).cpu().numpy().astype('float32')
		y=img2[i][:].squeeze(0).cpu().numpy().astype('float32')
		pred_mean=np.mean(x)
		dsm_mean=np.mean(y)
		pred_std=np.std(x,ddof=1)
		dsm_std=np.std(y,ddof=1)



		zncc = ((x - pred_mean) * (y - dsm_mean)) / (pred_std * dsm_std)
		# print(np.mean(zncc))
		# print(zncc.shape)
		# print(zncc.mean())
		# print(x.reshape(-1,1).shape)
		# print(x.reshape(-1,1).shape)
		# c = pearsonr(x.reshape(-1,1),y.reshape(-1,1))
		# print(c[0])
		# print(c[0])
		# print(a.shape)
		# print(a.mean())
		# print(np.corrcoef(x, y))
		zncc_list.append(zncc)
		# ase=mean_squared_error(x, y)
		# rmse=np.sqrt(mean_squared_error(x, y))
		# rmse_list.append(rmse)
	# print(np.mean(zncc_list))

	return np.mean(zncc_list)




if __name__ == '__main__':
	
	a=torch.randn((4,1,512,512))
	b=torch.randn((4,1,512,512))
	result=cal_mre(a,b)
	print(result)
	# a=np.array((4,1,512,512))
	# b=np.array((4,1,512,512))
	# print(cal_psnr(a,b))
	# print(cal_ssim(a,b))
	# print(cal_mae(a,b))
	# print(cal_rmse(a,b))
	# cal_zncc(a,b)