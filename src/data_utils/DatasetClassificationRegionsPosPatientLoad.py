import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from glob import glob
import os,sys
import SimpleITK as sitk
import pandas as pd
import random
from sklearn.model_selection import StratifiedGroupKFold
import itertools
def find_indices(list_to_check, item_to_find):
    array = np.array(list_to_check)
    indices = np.where(array == item_to_find)[0]
    return list(indices)

def torch_convert(img,transform=None):
    
    temp = torch.from_numpy(img) 
    #temp=temp[np.newaxis,...]
    temp=temp.unsqueeze(0)
    #temp=temp.unsqueeze(0)
    res=temp
    if transform!=None:
        #print("mphka")
        res = transform(temp) # mono tensor vale edw
        #res=temp['image']
        
    #res=temp['image']
    #print(res.shape)
    return res

sys.path.insert(0,'../')
#from dataset_utils import extract_sample_point_from_region
import time

def create_two_random_list_of_size_n(n):
    list1 = np.random.randint(1,10, size=n)
    list2 = np.random.randint(20,30, size=n)
    return list1, list2

def take_pairs_of_random_elements_from_two_lists(list1, list2, num_pairs):
    # get the number of elements in the list
    num_elements_l1 = len(list1)
    num_elements_l2 = len(list2)
    list1=np.array(list1)
    list2=np.array(list2)
    # get the random indexes
    random_indexes_l1 = np.array(np.random.randint(num_elements_l1, size=num_pairs)).astype(int)
    random_indexes_l2 = np.array(np.random.randint(num_elements_l2, size=num_pairs)).astype(int)
    # get the random pairs
    #l1 = [word for ]
    random_pairs = np.array([list1[random_indexes_l1], list2[random_indexes_l2]])
    return random_pairs

def take_combinations_of_random_elements_from_list(list1,num_pairs, num_comb):
    # get the number of elements in the list
    num_elements_l1 = len(list1)
    #num_elements_l2 = len(list2)
    list1=np.array(list1)
    #list2=np.array(list2)
    # get the random indexes
    random_indexes=[]
    for i in range(num_pairs):
        random_indexes.append(np.array(np.random.randint(num_elements_l1, size=num_comb)).astype(int))

    # get the random pairs
    #l1 = [word for ]
    random_pairs=[]
    for i in range(num_pairs):
        #random_pairs.append([list1[random_indexes[i][0]], list1[random_indexes[i][1]],list1[random_indexes[i][2]]])
        temp=[]
        for j in range(num_comb):
            temp.append(list1[random_indexes[i][j]])
        random_pairs.append(temp)

    random_pairs = np.array(random_pairs)
    print(random_pairs.shape)
    return random_pairs

class DatasetClassificationRegionsPosPatientLoad(Dataset):
    
    def __init__(self,data_list,transform,resize):
        self.data_list=data_list
        self.n_samples = len(self.data_list)
        self.resize=resize
        self.transform = transform
        if self.resize!=None:
            self.transform_augment = self.transform
    def __getitem__(self,index):
        
        cur_batch = self.data_list[index]
        loc = cur_batch[2]
        cl = cur_batch[1]
        img = cur_batch[0]
        if self.resize!=None:
            img = self.transform_augment(img)
        views=[img]
        y_list=[cl]
        loc_list=[loc]

        return views,y_list,loc_list


    def __len__(self):
        return self.n_samples