#DatasetPETCTContrastiveLOAD


import torchvision
from torch.utils.data import Dataset, DataLoader
from glob import glob
import sys
import pandas as pd
import random
from monai.transforms import RandRotate,RandFlip,RandAffine,RandCropByLabelClassesd,AsDiscreted,KeepLargestConnectedComponentd,Activations,Resized,ScaleIntensityRanged,ThresholdIntensityd,RandCropByPosNegLabeld,ToTensord ,Spacingd,EnsureChannelFirstd,AsDiscrete
import math
#import dataset_utils
import itertools

sys.path.insert(0,'../')

def repeat_elements(lst, d):
    n = len(lst)
    repetitions = d // n
    remainder = d % n
    result = lst * repetitions + lst[:remainder]
    return result

class DatasetClassificationRegionsPosPatientLoadAug(Dataset):
    
    def __init__(self,data_list,resize,transform,transform_augment=None,n_examples=1500):
        self.data_list=data_list
        self.transform_augment=transform_augment
        self.transform = transform
        self.resize=resize
        if self.resize!=None:
            self.transform_augment = torchvision.transforms.Compose([self.transform_augment,self.transform])

        self.indexes_0 = [i for i, x in enumerate(self.data_list) if x[1] == 0]
        self.indexes_1 = [i for i, x in enumerate(self.data_list) if x[1] == 1]
        
        random.shuffle(self.indexes_0)
        
        n_indexes_1 = len(self.indexes_1)
        self.indexes_0 = self.indexes_0[:int(n_examples*1.3)] #1.3
        self.indexes_1 = repeat_elements(self.indexes_1,n_examples)
        self.indexes = self.indexes_0 + self.indexes_1
        random.shuffle(self.indexes)
        self.n_samples = len(self.indexes)        
        self.prob_aug = 1 - n_indexes_1 / len(self.indexes_1) # ratio real samples/copies
        
    
    def __getitem__(self,index):
        
        index_data = self.indexes[index]
        cur_batch = self.data_list[index_data]
        loc = cur_batch[2]
        cl = cur_batch[1]
        img = cur_batch[0]
        
        if cl == 0:
            views=[]
            views.append(img)
        else:
            prob =random.random()
            if prob>self.prob_aug:
                views=[]
                views.append(img)
            else:
                views=[]
                views.append(self.transform_augment(img))
        
        y_list=[cl]        
        loc_list=[loc]

        return views,y_list,loc_list


    def __len__(self):
        return self.n_samples