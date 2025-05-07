import torch
import torchvision
import torch.nn as nn
import os, sys
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import Resize, ToTensord
from monai.transforms import (
    RandRotate,
    RandFlip,
    RandAffine,
    RandHistogramShift,
    RandAdjustContrast,
    RandGaussianNoise,
    ToTensord,
)
import argparse
import wandb
from lightly.utils.scheduler import CosineWarmupScheduler

from VICRegPosEncBlock import VICRegPosEncBlock
from VICRegPosEncBlock import VICRegPosEncBlock_VIT
from utils import LARS, exclude_bias_and_norm
from networks.resnet_model import resnet18, resnet10
from networks.efficientnet import EfficientNet3D


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


wandb.init(project="Representation learning sem-pos")

from utils import load_config_module

if __name__ == "__main__":
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    sys.path.insert(0, src_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="vicreg params")
    parser.add_argument(
        "--model_name", default="resnet18", help="model_name (default: resnet18)"
    )
    parser.add_argument(
        "--config", default="config.py",type=str, help="configuration file"
    )
    
    args = parser.parse_args()
    model_name = args.model_name
    config_file = load_config_module(args.config)
    
    
    """MAIN_PATH = "/home/tpv/followups/trainset_v2.csv"
    MAIN_PATH_VALID = "/home/tpv/followups/valset_v2.csv"
    PATH_LABELS = "../data_classification/patient_data_fixed_voxel_number.csv"""
    # HYPERPARAMETERS
    MAIN_PATH = config_file.MAIN_PATH
    MAIN_PATH_VALID = config_file.MAIN_PATH_VALID
    PATH_LABELS = config_file.PATH_LABELS
    MODELS_FOLDER = config_file.MODELS_FOLDER

    #check if MODELS_FOLDER exist
    if not os.path.exists(MODELS_FOLDER):
        try:
            os.makedirs(MODELS_FOLDER)
        except Exception as e:
            print(f"Error creating directory {MODELS_FOLDER}: {e}")
            sys.exit(1)
            
    # CONFIGURATION
    name = config_file.name
    mode = config_file.mode
    init_epoch = config_file.init_epoch
    num_epochs = config_file.num_epochs
    spatial_size = config_file.spatial_size
    batch_size = config_file.batch_size
    num_sub_vol = config_file.num_sub_vol
    learning_rate = config_file.learning_rate
    loc_size = config_file.loc_size
    
    #to be deleted
    print(MAIN_PATH)
    print(PATH_LABELS)
    
    # pairs_state: variable to control if semantic sampling is used
    pairs_state = True if mode == "pairs" else False
    resize_state = None
    if "efficientnet" in model_name:
        resize_state = True

    # MODELS PATH
    config = "_".join(["final", name, model_name, mode, "loc9"])  # loc9_perc10b
    model_dir = os.path.join(MODELS_FOLDER, config)
    
    loaded_checkpoint = None
    transform = torchvision.transforms.Compose([ToTensord(keys=["image"])])
    transform_eff = torchvision.transforms.Compose(
        [
            Resize(spatial_size=[64, 64, 64])
        ]
    )

    if resize_state:
        transform = transform_eff

    transform_augment = torchvision.transforms.Compose(
        [
            RandAffine(prob=0.8),
            RandFlip(prob=0.8),
            RandRotate(prob=0.8, range_x=10.0, range_y=10.0, range_z=10.0),
            RandGaussianNoise(prob=0.8),
            RandAdjustContrast(prob=0.8),
        ]
    )

    data_info_train = {
        "MAIN_PATH": MAIN_PATH,
        "PATH_LABELS": PATH_LABELS,
        "transform": transform,
        "batch_size": batch_size,
        "spatial_size": spatial_size,
        "k_views": 2,
        "pairs_state": pairs_state,
        "transform_augment": transform_augment,
        "resize_state": resize_state,
        "transforms_bool": False,
        "epoch": init_epoch,
        "cur_epoch": 0,
        "epoch_step": 800,  # 3800, 150,#60,
        "num_workers": 4,
        "n_samples": 2000,
    }  # public htan 6000, twra tha ginei 2000

    data_info_val = {
        "MAIN_PATH": MAIN_PATH_VALID,
        "PATH_LABELS": PATH_LABELS,
        "transform": transform,
        "batch_size": batch_size,
        "spatial_size": spatial_size,
        "k_views": 2,
        "pairs_state": pairs_state,
        "transform_augment": transform_augment,
        "resize_state": resize_state,
        "transforms_bool": False,
        "epoch": init_epoch,
        "cur_epoch": 0,
        "num_workers": 4,
        "n_samples": 1000,
    }  # htan 2000

    if model_name == "efficientnet":
        backbone = EfficientNet3D.from_name(
            "efficientnet-b0", override_params={"num_classes": 2}, in_channels=1
        )
        num_ftrs = backbone._fc.in_features
        backbone._fc = Identity()
    elif model_name == "resnet18":
        backbone = resnet18(in_channels=1, sample_size=spatial_size)
        num_ftrs = backbone.fc.in_features
        backbone.fc = Identity()
    elif model_name == "resnet10":
        backbone = resnet10(in_channels=1, sample_size=spatial_size)
        num_ftrs = backbone.fc.in_features
        backbone.fc = Identity()
    elif model_name == "vit":
        from monai.networks.nets import ViT

        num_ftrs = 768
        backbone = ViT(
            in_channels=1,
            img_size=spatial_size,
            pos_embed="conv",
            classification=True,
            patch_size=spatial_size,
            hidden_size=num_ftrs,
        )
        backbone.classification_head = Identity()

    model = VICRegPosEncBlock(backbone, num_ftrs=num_ftrs, loc_size=loc_size)
    if "vit" in model_name:
        model = VICRegPosEncBlock_VIT(backbone, num_ftrs=num_ftrs, loc_size=loc_size)
    
    model.to(device)

    optimizer = LARS(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6,
        momentum=0.9,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )
    
    #scheduler = None
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=1200,
        max_epochs=60 * (6000 / batch_size),
        end_value=0.001,
    )

    if loaded_checkpoint:
        print("weights loaded")
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])


    isExist = os.path.exists(model_dir)
    if not isExist:
        os.makedirs(model_dir)

    from selfsupervision_trainers import trainer_vicreg_loc

    trainer_vicreg_loc(
        model,
        num_epochs,
        optimizer,
        scheduler,
        device,
        model_dir,
        init_epoch,
        data_info_train,
        data_info_val,
        wandb=wandb,
    )
