import torch
import torchvision
from torch.utils.data import Dataset
import os,sys
import random

#import dataset_utils
from itertools import combinations
sys.path.insert(0,'../')

def torch_convert(img,transform=None):
    
    temp = torch.from_numpy(img) 
    temp=temp.unsqueeze(0)
    res=temp
    if transform!=None:
        res = transform(temp)
        
    return res

def calc_expected_number_of_combinations(n):
    return n * (n - 1) / 2


class DatasetRegionsPosPatientLoad(Dataset):
    
    def __init__(self,data_list,resize,pairs,transform=None,transform_augment=None,spatial_size=[32,32,32],samples_to_get=100):
        self.data_list=data_list
        self.transform=transform
        self.transform_augment=transform_augment
        self.spatial_size=spatial_size
        self.resize=resize
        self.pairs=pairs
        self.samples_to_get=samples_to_get
        if self.resize!=None:
            self.transform_augment = torchvision.transforms.Compose([self.transform_augment,self.transform])
        

        self.indexes_0 = [i for i, x in enumerate(self.data_list) if x[1] == 0]
        self.indexes_1 = [i for i, x in enumerate(self.data_list) if x[1] == 1]

        self.combs_indexes_1 = list(combinations(self.indexes_1, 2))
        random.shuffle(self.combs_indexes_1)
        self.combs_indexes_0 = list(combinations(self.indexes_0, 2))
        random.shuffle(self.combs_indexes_0)

        # select the number of samples and apply constraints to the number of samples set by the user
        expected_number_of_combinations = calc_expected_number_of_combinations(len(self.combs_indexes_1)) + calc_expected_number_of_combinations(len(self.combs_indexes_0))
        n_samples_temp = samples_to_get if samples_to_get < int(expected_number_of_combinations) else int(expected_number_of_combinations)

        # 80% of the data is index1
        # 20% of the data is index0
        self.indexes_combined = self.combs_indexes_1[:int(0.8 * n_samples_temp)] + self.combs_indexes_0[:int(0.2 * n_samples_temp)]
        #shuffle
        random.shuffle(self.indexes_combined)

        self.n_samples = len(self.indexes_combined)
        
    
    def __getitem__(self,index):

        cur_indexes = self.indexes_combined[index]
        # current indexes include region 1 and region 2
        # extact image, label and location for each region
        cur_batch_1 = self.data_list[cur_indexes[0]]
        loc_1 = cur_batch_1[2]
        cl_1 = cur_batch_1[1]
        img_1 = cur_batch_1[0]

        cur_batch_2 = self.data_list[cur_indexes[1]]
        loc_2 = cur_batch_2[2]
        cl_2 = cur_batch_2[1]
        img_2 = cur_batch_2[0]
        loc_views=[]
        views=[]
        if self.resize==None:
            views.append(torch_convert(img_1))
        else:
            views.append(torch_convert(img_1,self.transform))
        
        loc_views.append(loc_1)

        # self.pairs regulate the use of semantic sampling, if True, the semantic sampling is used else only augmentations are used
        if self.pairs:
            prob = random.random()
            if prob>0.2:
                views.append(torch_convert(img_2,self.transform_augment))
                loc_views.append(loc_2)
            else:
                views.append(torch_convert(img_1,self.transform_augment))
                loc_views.append(loc_1)
        else:
            views.append(torch_convert(img_1,self.transform_augment))
            loc_views.append(loc_1)
        
        return views,loc_views


    def __len__(self):
        return self.n_samples