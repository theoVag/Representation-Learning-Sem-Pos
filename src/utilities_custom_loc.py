#!/usr/bin/env python3
# Data loading utilities for position encoding experiments: Representation Learning and Classification

import torch
from data_utils.DatasetRegionsPosPatient import DatasetRegionsPosPatient
from data_utils.DatasetRegionsPosPatientLoad import DatasetRegionsPosPatientLoad
from data_utils.DatasetClassificationRegionsPosPatientLoad import DatasetClassificationRegionsPosPatientLoad
from data_utils.DatasetClassificationRegionsPosPatient import DatasetClassificationRegionsPosPatient
from data_utils.DatasetClassificationRegionsPosPatientLoadAug import DatasetClassificationRegionsPosPatientLoadAug

import sys
sys.path.insert(0,'../')
def create_loader(data_info):
    train_regions_samples = DatasetRegionsPosPatient(MAIN_PATH=data_info['MAIN_PATH'],spatial_size=data_info['spatial_size'],PATH_LABELS=data_info['PATH_LABELS'])
    regions_train=[]
    for batch in train_regions_samples:
        regions_train+=batch
    
    residual = len(regions_train) % data_info['batch_size']
    regions_train = regions_train[:(len(regions_train)-residual)]
    train_dataset = DatasetRegionsPosPatientLoad(data_list = regions_train,resize=data_info['resize_state'],pairs=data_info['pairs_state'],transform=data_info['transform'],transform_augment=data_info['transform_augment'],spatial_size=data_info['spatial_size'],samples_to_get = data_info['n_samples'])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=data_info['batch_size'],shuffle=True,num_workers=data_info['num_workers']) #1
    
    return train_dataset,train_loader


def create_loader_classification(data_info):
    train_regions_samples = DatasetClassificationRegionsPosPatient(MAIN_PATH=data_info['MAIN_PATH'],spatial_size=data_info['spatial_size'],PATH_LABELS=data_info['PATH_LABELS'],train_phase=True)

    regions_train=[]
    for batch in train_regions_samples:
        bb = [(batch[0][i],batch[1][i],batch[2][i]) for i in range(len(batch[0]))]
        regions_train+=bb
    
    residual = len(regions_train) % data_info['batch_size']
    regions_train = regions_train[:(len(regions_train)-residual)]

    if 'val' in data_info['MAIN_PATH']:
        train_dataset = DatasetClassificationRegionsPosPatientLoad(data_list = regions_train,resize=data_info['resize_state'],transform=data_info['transform'],transform_augment=data_info['transform_augment'])
        shuffle=False
    else:
        train_dataset = DatasetClassificationRegionsPosPatientLoadAug(data_list = regions_train,resize=data_info['resize_state'],transform=data_info['transform'],transform_augment=data_info['transform_augment'],n_examples=900) #,n_examples=900
        shuffle=True

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=data_info['batch_size'],shuffle=shuffle,num_workers=4) #1
    return train_dataset,train_loader
