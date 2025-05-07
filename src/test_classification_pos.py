import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
import os
import pandas as pd
from monai.transforms import ToTensord,Resize,RandRotate,RandFlip,RandAffine,RandHistogramShift,RandAdjustContrast,RandGaussianNoise
from sklearn.metrics import confusion_matrix,classification_report

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
sys.path.insert(0, '../')
from networks.resnet_model import resnet18,resnet10,resnet50
from networks.efficientnet import EfficientNet3D
from networks.class_models import PosEncBlockClass, ClassLinearHead
from data_utils.DatasetClassificationRegionsPosPatient import DatasetClassificationRegionsPosPatient
import argparse
from utils import load_config_module

if __name__== "__main__":
    
    # HYPERPARAMETERS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes=2
    init_epoch=0
    num_classes=1
    num_epochs = 800
    spatial_size = [16,16,16]
    loc_size=9
    SAVE_DIR = '../classification_results/'
    MAIN_PATH = "dataset_simulate.csv" #csv with the volume file paths
    PATH_LABELS = 'patients_labels.csv'
    MODELS_FOLDER = '../models_class/'

    #check if SAVE_DIR exist
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
        except Exception as e:
            print(f"Error creating directory {SAVE_DIR}: {e}")
            sys.exit(1)
    
    resize_state=None
    
    
    parser = argparse.ArgumentParser(description='vicreg params')
    #parser.add_argument('--model_name', default='VICREG_LOC', help='model_name VICREG')
    parser.add_argument('--backbone_name', default='resnet18', help='backbone_name resnet18')
    parser.add_argument(
        "--config", default="config_classification.py",type=str, help="configuration file"
    )
    args = parser.parse_args()
    #name = args.model_name
    backbone_name = args.backbone_name
    config_file = load_config_module(args.config)
    
    
    # HYPERPARAMETERS
    init_epoch = config_file.init_epoch
    num_classes = config_file.num_classes
    num_epochs = config_file.num_epochs
    spatial_size = config_file.spatial_size
    loc_num_boxes = config_file.loc_num_boxes
    num_sub_vol = config_file.num_sub_vol
    learning_rate = config_file.learning_rate
    batch_size = config_file.batch_size
    loc_size = config_file.loc_size
    FINETUNE = config_file.FINETUNE

    # CONFIGURATION
    name = config_file.name
    mode = config_file.mode
    
    # PATHS
    MAIN_PATH = config_file.MAIN_PATH_VALID
    PATH_LABELS = config_file.PATH_LABELS
    MODELS_FOLDER = config_file.MODELS_FOLDER
    #MODELS_FOLDER_PRET=config_file.MODELS_FOLDER_PRET
    
    
    config = "_".join(['final',name,backbone_name,mode,'loc9'])
    model_dir = os.path.join(MODELS_FOLDER, config)
    model_path = os.path.join(model_dir, 'best_model_checkpoint.pth')
    
    transform = torchvision.transforms.Compose([ToTensord(keys=["image"])]) 
    transform_eff = torchvision.transforms.Compose([
                                                Resize(spatial_size=[64,64,64])                                               
                                                ])
    transform_augment = torchvision.transforms.Compose([RandFlip(prob=0.8),RandRotate(prob=0.8,range_x=10.0,range_y=10.0,range_z=10.0),RandGaussianNoise(prob=0.8),RandHistogramShift(prob=0.8)])

    # Resize for efficientnet
    resize_state=None
    if 'efficientnet' in backbone_name:
        resize_state = True
        transform=transform_eff
    
    
    if 'resnet50' in backbone_name:
        backbone = resnet50(in_channels=1, sample_size=spatial_size)
        num_ftrs=backbone.fc.in_features
        backbone.fc = Identity()
        model = PosEncBlockClass(backbone,num_ftrs=num_ftrs,loc_size=loc_size)
    elif 'resnet18' in backbone_name:
        backbone = resnet18(in_channels=1, sample_size=spatial_size)
        num_ftrs=backbone.fc.in_features
        backbone.fc = Identity()
        model = PosEncBlockClass(backbone,num_ftrs=num_ftrs,loc_size=loc_size)
    elif 'resnet10' in backbone_name:
        backbone = resnet10(in_channels=1, sample_size=spatial_size)
        num_ftrs=backbone.fc.in_features
        backbone.fc=Identity()
        model = PosEncBlockClass(backbone,num_ftrs=num_ftrs,loc_size=loc_size)
    elif 'senet' in backbone_name:
        from monai.networks.nets import SEResNet50
        backbone = SEResNet50(spatial_dims=3,in_channels=1,layers=(3, 4, 6, 3), groups=4, reduction=16, dropout_prob=None, inplanes=64,num_classes=512)
        backbone.logits=Identity()
        num_ftrs=512
        model = PosEncBlockClass(backbone,num_ftrs=num_ftrs,loc_size=loc_size)
    elif 'efficientnet' in backbone_name:
        backbone=EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)
        num_ftrs = backbone._fc.in_features
        dim_in=num_ftrs
        backbone._fc=Identity()
        model = PosEncBlockClass(backbone,num_ftrs=num_ftrs,loc_size=loc_size)
    elif 'vit' in backbone_name:
        from monai.networks.nets import ViT
        dim_in=768
        num_ftrs=768
        model = ViT(in_channels=1, img_size=spatial_size, pos_embed='conv', classification=True,patch_size=spatial_size,hidden_size=dim_in)        
        model.classification_head = Identity()
        from networks.class_models import PosEncBlockClass_vit
        model = PosEncBlockClass_vit(model,num_ftrs=dim_in,loc_size=loc_size)
    
    
    model_full = ClassLinearHead(model,in_features=num_ftrs,num_classes=2) # sbhse ena layer edw na meinei to mono 
    model_full.to(device)  
    
    dim_in=num_ftrs


    model_dir = model_path
       
    print(model_dir)
    
    loaded_checkpoint = torch.load(model_dir)
    

    transform = torchvision.transforms.Compose([ToTensord(keys=["image"])]) 

    if 'efficientnet' in backbone_name:
        resize_state=True
        transform_eff= torchvision.transforms.Compose([Resize(spatial_size=[64,64,64])])

    transform_augment = torchvision.transforms.Compose([RandAffine(prob=0.8), RandFlip(prob=0.8),RandRotate(prob=0.8,range_x=10.0,range_y=10.0,range_z=10.0),RandGaussianNoise(prob=0.8),RandAdjustContrast(prob=0.8),RandHistogramShift(prob=0.8)])
    
    test_dataset = DatasetClassificationRegionsPosPatient(MAIN_PATH=MAIN_PATH,spatial_size=spatial_size,PATH_LABELS=PATH_LABELS)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1)     
    
    
    model_full.load_state_dict(loaded_checkpoint["model_state_dict"],strict=True)
    
    
    # validation step 
    test_steps = len(test_loader)
    metric_values=[]
    n_class=2
    cm_list=[]
    f1_list=[]
    recall_list=[]
    precision_list=[]
    features_full=[]
    y_features=[]
    y_test_plot=[]
    y_pred_plot=[]
    model_full.eval()
    
    with torch.no_grad():
        step=0
        for i, batch in enumerate(test_loader):
            images,y,loc = batch#[0]

            y_hat_list = []
            y_test_list = []

            images = torch.cat(images, 0)
            y = torch.cat(y, 0)
            loc = torch.cat(loc, 0)
            step+=1
            images, y, loc = images.to(device), y.to(device), loc.to(device)
            y=y.float()
            batch_dim = y.shape[0]
            if images.shape[0]==1:
                continue

            #apply transform to images            
            if resize_state:
                images = torch.stack([transform_eff(im) for im in images])
            outputs = model_full(images,loc)
            y = torch.reshape(y, (len(y),1))
            y_test = y.cpu()
            y_hat = outputs.cpu()
            y_pred_plot.append(y_hat[:,1].numpy())
            y_hat = torch.argmax(y_hat,dim=1)

            y_test_list.append(y_test.numpy())
            y_hat_list.append(y_hat.numpy())
            
            y_test_plot.append(y_test.numpy())
            y_hat_list = np.concatenate(y_hat_list)
            y_test_list = np.concatenate(y_test_list)  

            cm_list.append(confusion_matrix(y_test_list, y_hat_list,labels=[0,1]))
            print(cm_list[-1])
            cl_report = classification_report(y_test_list, y_hat_list,output_dict=True,labels=np.array([0,1]),zero_division=0)
            f_temp = [cl_report[str(word)]['f1-score'] for word in range(n_class)]
            print(f_temp)
            recall_temp = [cl_report[str(word)]['recall'] for word in range(n_class)]
            precision_temp = [cl_report[str(word)]['precision'] for word in range(n_class)]
            f1_list.append(f_temp)
            recall_list.append(recall_temp)
            precision_list.append(precision_temp)


    final_cm = np.sum(cm_list,axis=0)
    
    categories = ['notumor','tumor']
    df_cm = pd.DataFrame(final_cm, 
                         index = [ 'notumor', 'tumor'],
                         columns = ['notumor', 'tumor'])
    
    TP = np.diag(final_cm)
    FP = np.sum(final_cm, axis=0) - TP
    FN = np.sum(final_cm, axis=1) - TP
    num_classes = 2
    TN = []
    for i in range(num_classes):
        temp = np.delete(final_cm, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    f1score = (2*precision*recall)/(precision+recall)

    cm_full =np.sum(cm_list,axis=0)
    ratio_tumors = (cm_full[1][0]+cm_full[1][1])/(np.sum(cm_full))
    ratio_notumors = (cm_full[0][0]+cm_full[0][1])/(np.sum(cm_full))

    import dataframe_image as dfi
    from sklearn.metrics import roc_curve, auc

    from metrics_calculations import plot_auc,calculate_metrics_from_cm

    accuracy = (cm_full[0][0]+cm_full[1][1])/(np.sum(cm_full))
    acc_class, precision, recall, f1score, specificity = calculate_metrics_from_cm(cm_full)
    balanced_accuracy = (recall + specificity)/2
    print("balanced accuracy ", balanced_accuracy)
    
    y_pred = y_pred_plot
    y_test=y_test_plot
    y_pred = np.concatenate(y_pred)
    y_test = np.concatenate(y_test)
    y_pred = y_pred.tolist()
    y_test=[int(word[0]) for word in y_test]
    pred_data = pd.DataFrame({'y_hat_proba':y_pred,'Class':y_test})
    pred_data.to_csv(os.path.join(SAVE_DIR,'preds_data_'+config+'.csv'),index=False)
    # load class_df
    class_df = pd.read_csv('class_df.csv')
    if 'private' in MAIN_PATH:
        class_df = class_df[['Patient','Label','Class','isBrain','isKidney','isHeart','isBladder']]
    else:
        class_df = class_df[['Patient','Label','Class','ratio','isBrain','isKidney','isHeart','isBladder']]
    class_df['y_hat_proba'] = y_pred
    y_pred2 = [0 if word<0.5 else 1 for word in y_pred]
    class_df['y_hat'] = y_pred2
    class_df['y_test'] = y_test

    
    class_df.to_csv(os.path.join(SAVE_DIR,'class_df_'+config+'.csv'))

    path_metric = SAVE_DIR + name+'_'+backbone_name + '.png'
    fpr1,tpr1,thresholds1, roc_auc_model1 = plot_auc(y_pred,y_test,path_fig=SAVE_DIR)
    
    cols = ['name','backbone','model_path','cm','Specificity_0','Specificity_1', 'Precision_0','Precision_1','Recall_0','Recall_1', 'f1-score_0','f1-score_1','Accuracy','Balanced_Accuracy','AUC_Area']

    lst2 = [name,backbone_name,model_dir,cm_full,specificity[0],specificity[1],  precision[0],precision[1],recall[0],recall[1],f1score[0],f1score[1],accuracy,balanced_accuracy[0],roc_auc_model1]
    
    if not os.path.exists(os.path.join(SAVE_DIR,'loc_classification_results.csv')):
        files_save = pd.DataFrame([lst2], columns=cols)
        files_save.to_csv(os.path.join(SAVE_DIR,'loc_classification_results.csv'),index=False)
    else:
        #open in append mode
        files_save = pd.read_csv(os.path.join(SAVE_DIR,'loc_classification_results.csv'))
        row = pd.DataFrame([lst2],columns=cols)
        files_save = pd.concat([files_save, row], ignore_index=True)

    files_save.to_csv(os.path.join(SAVE_DIR,'loc_classification_results.csv'),index=False)