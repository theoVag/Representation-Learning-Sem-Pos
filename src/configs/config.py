# HYPERPARAMETERS
MAIN_PATH = "dataset_simulate.csv"
MAIN_PATH_VALID = "dataset_simulate.csv"
PATH_LABELS = 'patients_labels.csv'
MODELS_FOLDER = '../models'
# CONFIGURATION
name = "VICREG_LOC"
mode = "pairs"    
init_epoch = 0
num_epochs = 8 #800  # 800
spatial_size = [16, 16, 16]
batch_size = 4#128
num_sub_vol = 48
learning_rate = 0.05  # 0.005
loc_size = 9