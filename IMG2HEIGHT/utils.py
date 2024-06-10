from skimage import transform
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Test_Nptranspose(object):
    def __call__(self, sample):
        image = sample["image"]
        # label = sample["label"]

        image = image.transpose(2, 0, 1)
        # label = label.transpose(2,0,1)

        sample["image"] = image
        # sample["label"] = label

        return sample


class Nptranspose(object):
    def __call__(self,sample):
        image = sample["image"]
        label = sample["label"]
        
        image = image.transpose(2,0,1)
        label = label.transpose(2,0,1)
        
        sample["image"] = image
        sample["label"] = label
        
        return sample

class Rotation(object):
    def __init__(self,angle=90):
        self.angle = angle
    def __call__(self,sample):
        image,label = sample["image"],sample["label"]
        ids = np.around(360/self.angle)
        multi = np.random.randint(0,ids)
        if multi>0.001:
            # transform.rotate will change the range of the value
            image = transform.rotate(image,self.angle*multi).astype(np.float32)
            label = transform.rotate(label,self.angle*multi).astype(np.float32)
            sample["image"] = image
            sample["label"] = label
        return sample

class H_Mirror(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        image, label = sample["image"], sample["label"]
        
        if np.random.random()<self.p:
            new_image = np.flip(image,0).copy()
            new_label = np.flip(label,0).copy()
            return {'image':new_image, 'label':new_label}
        else:
            return {'image':image, 'label':label}

class V_Mirror(object):
    def __init__(self,p = 0.5):
        self.p = p
    def __call__(self,sample):
        image, label = sample["image"], sample["label"]
        
        if np.random.random()<self.p:
            new_image = np.flip(image,1).copy()
            new_label = np.flip(label,1).copy()
            return {'image':new_image, 'label':new_label}
        else:
            return {'image':image, 'label':label}