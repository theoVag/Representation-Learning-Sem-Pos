import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
import os,sys
from monai.transforms import ToTensord
from monai.transforms import Resize,RandRotate,RandFlip,RandHistogramShift,RandGaussianNoise,ToTensord
import wandb
import argparse
from networks.resnet_model import resnet18,resnet10,resnet50
from losses import FocalLoss
from networks.class_models import PosEncBlockClass, ClassLinearHead
from trainer_classification_loc import trainer_classification_pos
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
from networks.efficientnet import EfficientNet3D
from utils import load_config_module
wandb.init(project='pet-classification')


if __name__== "__main__":
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    sys.path.insert(0, src_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #HYPERPARAMETERS
    init_epoch=0
    num_classes=1
    num_epochs = 800#900# 600
    spatial_size = [16,16,16]
    loc_num_boxes=spatial_size
    num_sub_vol= 48
    learning_rate = 0.00001 
    batch_size = 64
    name='vicreg'
    loc_size=9
    mode = 'pairs'
    FINETUNE=False
    """MAIN_PATH = "/home/tpv/followups/trainset_v2.csv"
    MAIN_PATH_VALID = "/home/tpv/followups/valset_v2.csv"
    PATH_LABELS = '../data_classification/patient_data_fixed_voxel_number.csv'
    MODELS_FOLDER = '../models_loc_final'"""
    
    MAIN_PATH = "dataset_simulate.csv"
    MAIN_PATH_VALID = "dataset_simulate.csv"
    PATH_LABELS = 'patients_labels.csv'
    MODELS_FOLDER = '../models_class'
    
    

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
    MAIN_PATH = config_file.MAIN_PATH
    MAIN_PATH_VALID = config_file.MAIN_PATH_VALID
    PATH_LABELS = config_file.PATH_LABELS
    MODELS_FOLDER = config_file.MODELS_FOLDER
    MODELS_FOLDER_PRET=config_file.MODELS_FOLDER_PRET
    
    #check if MODELS_FOLDER exist
    if not os.path.exists(MODELS_FOLDER):
        try:
            os.makedirs(MODELS_FOLDER)
        except Exception as e:
            print(f"Error creating directory {MODELS_FOLDER}: {e}")
            sys.exit(1)
    
    # SAVE PATH FOR THE MODEL
    pairs_state = True if mode == 'pairs' else False
    config = "_".join(['final',name,backbone_name,mode,'loc9']) #loc9_perc10b
    model_dir = os.path.join(MODELS_FOLDER, config)
    model_path = os.path.join(os.path.join(MODELS_FOLDER_PRET, config), 'best_model_checkpoint.pth') 


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


    data_info_train = {"MAIN_PATH": MAIN_PATH,
                 "PATH_LABELS":PATH_LABELS,
                 "transform": transform,
                 "batch_size":batch_size*2,
                 "spatial_size":spatial_size,
                 "pairs_state":pairs_state,
                 "transform_augment": transform_augment,
                 "resize_state":resize_state,
                 "epoch":0,
                 "epoch_step":400,
                 "num_workers":8}

    data_info_val = {"MAIN_PATH": MAIN_PATH_VALID,
                 "PATH_LABELS":PATH_LABELS,
                 "transform": transform,
                 "batch_size":batch_size,
                 "spatial_size":spatial_size,
                 "pairs_state":pairs_state,
                 "transform_augment": transform_augment,
                 "resize_state":resize_state,
                 "epoch":0,
                 "num_workers":8}
    
    dim_in = 512

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
    
    print(model_path)
    state_dict = torch.load(model_path)
    state_dict = state_dict["model_state_dict"]

    if not FINETUNE:
        print("FROZEN WEIGHTS")
        for param in model.parameters():
            param.requires_grad = False
    
    
    model_full = ClassLinearHead(model,in_features=num_ftrs,num_classes=2) # sbhse ena layer edw na meinei to mono 
    model_full.to(device)  
    
    optimizer = torch.optim.Adam(model_full.parameters(), lr=learning_rate)    
    criterion = FocalLoss(alpha=0.55,gamma=2)
    
    model_dir = model_dir.replace('best_model_checkpoint.pth','')
    print(model_dir)
    
    scheduler=None
    isExist = os.path.exists(model_dir)
    if not isExist:
        os.makedirs(model_dir)
    
    trainer_classification_pos(model_full, num_epochs, criterion,optimizer,device,model_dir,init_epoch,data_info_train,data_info_val,wandb=wandb)
    

    
    
    
