#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        #CE_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        F_loss = F_loss.mean() # added then
        return F_loss

class BCEweighted(torch.nn.Module):
    
    def __init__(self, w_p = 8, w_n = 0.5):
        super(BCEweighted, self).__init__()
        
        self.w_p = w_p
        self.w_n = w_n
        
    def forward(self, logits, labels, epsilon = 1e-7):
        #ps = torch.sigmoid(logits.squeeze()) 
        ps=logits
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
        loss = loss_pos + loss_neg
        
        return loss
