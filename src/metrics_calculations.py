#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:37:27 2024

@author: tpv
"""
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import dataframe_image as dfi
from sklearn.metrics import roc_curve, auc
import matplotlib
SMALL_SIZE=18
BIGGER_SIZE=18
matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)
def plot_auc(y_pred,y_test,path_fig="plots"):
    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred)
    roc_auc_model1 = auc(fpr1, tpr1)
    print("AUC AREA ",roc_auc_model1)
    if path_fig!=None:

        plt.figure(figsize = (10,8))
        plt.plot(fpr1, tpr1,marker='.', label='Model (area = %0.2f)' % roc_auc_model1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title('Receiver Operating Characteristic Curve - AUC = %0.2f' % roc_auc_model1, fontsize=21)
        plt.savefig(path_fig + 'roc_auc.png',dpi=200)
    return fpr1,tpr1,thresholds1, roc_auc_model1

def calculate_metrics_from_cm(final_cm,num_classes = 2):
    TP = np.diag(final_cm)
    FP = np.sum(final_cm, axis=0) - TP
    FN = np.sum(final_cm, axis=1) - TP
    
    TN = []
    for i in range(num_classes):
        temp = np.delete(final_cm, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    f1score = (2*precision*recall)/(precision+recall)
    acc_class = final_cm.diagonal()/final_cm.sum(axis=1)
    
    full_acc = (TP+TN)/(TP+TN+FP+FN)
    print("Accuracy per class ",acc_class)
    print("precision{}".format(precision))
    print("recall{}".format(recall))
    print("specificity{}".format(specificity))
    print("f1_score{}".format(f1score))
    print("FULL ACCURACY {}".format(full_acc))
    return acc_class, precision, recall, f1score, specificity