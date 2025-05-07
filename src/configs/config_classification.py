# HYPERPARAMETERS

init_epoch=0
num_classes=1
num_epochs = 8#800#900# 600
spatial_size = [16,16,16]
loc_num_boxes=spatial_size
num_sub_vol= 48
learning_rate = 0.00001 
batch_size = 4#64
name = "VICREG_LOC"
loc_size=9
mode = 'pairs'
FINETUNE=False

MAIN_PATH = "dataset_simulate.csv"
MAIN_PATH_VALID = "dataset_simulate.csv"
PATH_LABELS = 'patients_labels.csv'
MODELS_FOLDER = '../models_class'
MODELS_FOLDER_PRET = '../models'