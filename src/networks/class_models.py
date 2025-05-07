#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:18:53 2024

@author: tpv
"""
from torch import nn    
import torch

class ClassLinearHead(nn.Module):
    def __init__(self,backbone,in_features, num_classes=1):
        super().__init__()
        self.backbone=backbone
        self.layer1 = nn.Linear(in_features, 128)
        self.layer2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Softmax(dim=1)
        self.act =nn.GELU()
 
    def forward(self, x,loc):
        x=self.backbone(x,loc)
        x=self.act(self.layer1(x))  
        x = self.sigmoid(self.layer2(x))
        return x

   
class PosEncBlockClass(nn.Module):
    def __init__(self,backbone,num_ftrs=512,loc_size=9, num_classes=1):
        
        super().__init__()
        self.backbone=backbone
        num_params = loc_size
        self.trainable_parameters = nn.Parameter(torch.randn(1, num_params))
        self.loc_fc = nn.Linear(in_features = loc_size, out_features=num_ftrs)
        self.fc1 = nn.Linear(in_features = int(num_ftrs)+num_ftrs, out_features=num_ftrs)
 
    def forward(self, x,loc):
        x = self.backbone(x).flatten(start_dim=1)
        x = x.unsqueeze(1)
        x = nn.functional.normalize(x, p=2, dim=1)
        loc = self.trainable_parameters * loc
        loc = self.loc_fc(loc)
        loc = nn.functional.normalize(loc, p=2, dim=1)
        x = torch.squeeze(x)
        loc = torch.squeeze(loc)
        x = torch.concatenate((x, loc), axis=1)
        x = self.fc1(x)
        return x
    
class PosEncBlockClass_vit(nn.Module):
    def __init__(self,backbone,num_ftrs=512,loc_size=9, num_classes=1):
        
        super().__init__()
        self.backbone=backbone
        num_params = loc_size
        self.trainable_parameters = nn.Parameter(torch.randn(1, num_params))
        self.loc_fc = nn.Linear(in_features = loc_size, out_features=num_ftrs)
        self.fc1 = nn.Linear(in_features = int(num_ftrs)+num_ftrs, out_features=num_ftrs)
 
    def forward(self, x,loc):
        x = self.backbone(x)
        x=x[0].flatten(start_dim=1)
        x = x.unsqueeze(1)
        x = nn.functional.normalize(x, p=2, dim=1)
        loc = self.trainable_parameters * loc
        loc = self.loc_fc(loc)
        loc = nn.functional.normalize(loc, p=2, dim=1)
        x = torch.squeeze(x)
        loc = torch.squeeze(loc)
        x = torch.concatenate((x, loc), axis=1)
        x = self.fc1(x)
        return x
