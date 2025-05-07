#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:15:53 2023

@author: tpv
"""

import torch
from torch import nn

from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VICRegProjectionHead

class VICRegPosEncBlock(nn.Module):
    def __init__(self, backbone,num_ftrs,loc_size=9):
        super().__init__()
        self.backbone = backbone
        num_params = loc_size
        self.trainable_parameters = nn.Parameter(torch.randn(1, num_params))
        self.loc_fc = nn.Linear(in_features = loc_size, out_features=num_ftrs) # an balw thesh 128 enwsh me 512 + fc1128
        #self.projection_head = BarlowTwinsProjectionHead(num_ftrs, 2048, 2048)
        self.projection_head = VICRegProjectionHead(
            input_dim=num_ftrs,
            hidden_dim=4*num_ftrs,#2048,
            output_dim=4*num_ftrs,#2048,
            num_layers=2,
        )
        self.fc1 = nn.Linear(in_features = 2*num_ftrs, out_features=num_ftrs) # sto addition 1 fora
        
    def forward(self, x,loc):
        x = self.backbone(x).flatten(start_dim=1)
        x = nn.functional.normalize(x, p=2, dim=1)
        loc = self.trainable_parameters * loc
        loc = self.loc_fc(loc)
        loc = nn.functional.normalize(loc, p=2, dim=1)
        x = torch.concatenate((x, loc), axis=1)
        x = self.fc1(x)
        z = self.projection_head(x)
        return z
    
class VICReg(nn.Module):
    def __init__(self, backbone,num_ftrs):
        super().__init__()
        self.backbone = backbone
        
        self.projection_head = VICRegProjectionHead(
            input_dim=num_ftrs,
            hidden_dim=4*num_ftrs,
            output_dim=4*num_ftrs,
            num_layers=2,
        )
        
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = nn.functional.normalize(x, p=2, dim=1)
        
        z = self.projection_head(x)
        return z
    
class VICRegPosEncBlock_VIT(nn.Module):
    def __init__(self, backbone,num_ftrs,loc_size=9):
        super().__init__()
        self.backbone = backbone
        
        num_params = loc_size
        self.trainable_parameters = nn.Parameter(torch.randn(1, num_params))

        self.loc_fc = nn.Linear(in_features = loc_size, out_features=num_ftrs) # an balw thesh 128 enwsh me 512 + fc1128
        
        self.projection_head = VICRegProjectionHead(
            input_dim=num_ftrs,
            hidden_dim=4*num_ftrs,#2048,
            output_dim=4*num_ftrs,#2048,
            num_layers=2,
        )
        self.fc1 = nn.Linear(in_features = 2*num_ftrs, out_features=num_ftrs) # sto addition 1 fora
        
    def forward(self, x,loc):
        x = self.backbone(x)
        x = x[0].flatten(start_dim=1)
        x = nn.functional.normalize(x, p=2, dim=1)

        loc = self.trainable_parameters * loc
        loc = self.loc_fc(loc)
        loc = nn.functional.normalize(loc, p=2, dim=1)

        x = torch.concatenate((x, loc), axis=1)
        x = self.fc1(x) 
        z = self.projection_head(x)
        return z
