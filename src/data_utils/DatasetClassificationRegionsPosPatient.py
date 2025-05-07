
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os,sys
import SimpleITK as sitk
import pandas as pd
sys.path.insert(0,'../')
import warnings
import SimpleITK as sitk
import location_vector_utils

def find_centroid(label, labeled_array):
    indices = np.array(np.where(labeled_array == label))
    centroid = np.mean(indices, axis=1).astype(int)
    return centroid

class DatasetClassificationRegionsPosPatient(Dataset):
    
    def __init__(self,MAIN_PATH,
                 spatial_size=[32,32,32],
                 PATH_LABELS='../data_classification/patient_data_fixed_voxel_number.csv',
                 train_phase=False):
        
        self.MAIN_PATH=MAIN_PATH
        self.spatial_size=spatial_size
        self.PATH_LABELS=PATH_LABELS
        self.train_phase=train_phase
        #random.seed(1234)
        
        if MAIN_PATH.endswith(".csv"):
            fnames = pd.read_csv(MAIN_PATH,header=None)
            self.volumes_paths = fnames.iloc[:,0].tolist()
            self.segs_paths = [word.replace('VOLS','SEGS_MULTI') for word in self.volumes_paths]
            #self.segs_paths = [word.replace('SUV','SEG_LABELS').replace('.nii.gz','_multi_label.nii.gz') for word in self.volumes_paths] # to exw kanei me epikalupsh alla sto inhouse dataset exw kai kala ta labels

            self.labels_csv = pd.read_csv(PATH_LABELS)
            print("Data loading...")
            """if train_phase:
                voxel_threshold=85
                samples_p1 = self.labels_csv.loc[self.labels_csv["VoxelNumber"]<voxel_threshold,] #83 64

                #samples_p1 = self.labels_csv.loc[self.labels_csv["original_shape_VoxelVolume"]<64*16,]
                samples_p1_cl0 = samples_p1.loc[samples_p1["Class"]==0,]
                samples_p1_cl1 = samples_p1.loc[samples_p1["Class"]==1,]
                # sample random from p1_c0 20%
                samples_p1_cl0=samples_p1_cl0.sample(frac=0.9, replace=False) #0.9 to kalo 0.02
                samples_p2 = self.labels_csv.loc[self.labels_csv["VoxelNumber"]>=voxel_threshold,]

                samples_p2_cl0 = samples_p2.loc[samples_p2["Class"]==0,]
                samples_p2_cl1 = samples_p2.loc[samples_p2["Class"]==1,]
                samples_p2_cl0=samples_p2_cl0.sample(frac=0.9, replace=False) #0.1 #0.03
                self.labels_csv = pd.concat([samples_p1_cl0,samples_p1_cl1,samples_p2_cl0,samples_p2_cl1],axis=0)

            
            self.labels_csv = self.labels_csv[self.labels_csv['isBrain']!=1]
            self.labels_csv = self.labels_csv[self.labels_csv['isBladder']!=1]
            self.labels_csv = self.labels_csv[self.labels_csv['isKidney']!=1]
            self.labels_csv = self.labels_csv[self.labels_csv['isHeart']!=1]"""
            
            patients_list = [word.split('/')[-1].replace('.nii.gz','') for word in self.volumes_paths]
            self.labels_csv = self.labels_csv.loc[self.labels_csv['Patient'].isin(patients_list)]

        self.n_samples = len(self.volumes_paths)
        
    
    def __getitem__(self,index):

        dim=self.spatial_size[0]
        volume = sitk.ReadImage(self.volumes_paths[index])
        labels = sitk.ReadImage(self.segs_paths[index])
        volume = sitk.Cast(volume, sitk.sitkFloat32)
            
        #volume = sitk.IntensityWindowing(volume, 0, 40, 0, 40)
        volume=sitk.Normalize(volume)
        
        
        patient = self.volumes_paths[index].split('/')[-1].replace('.nii.gz','')
        cur_patient = self.labels_csv.loc[self.labels_csv['Patient']==patient]
        
        volume_numpy = sitk.GetArrayFromImage(volume)
        numpy_labels=sitk.GetArrayFromImage(labels)
        
        num_labels_1 = list(cur_patient.loc[:,'Label'])
        num_labels_2 = list(np.unique(numpy_labels))
        num_labels = list(set(num_labels_1) & set(num_labels_2))
        
        img_list=[]
        y_list=[]        
        list_loc_vector=[]

        for lab in num_labels:
            #tt=time.time()
            if lab==0:
                continue
            cur_class = cur_patient.loc[cur_patient['Label']==lab,'Class']

            try:
                cent=[find_centroid(lab,numpy_labels)]
                cent=cent[0]
                warnings.filterwarnings("error", category=RuntimeWarning)
            except RuntimeWarning as rw:
                print(f"Caught a runtime warning--: {rw}")
                continue

            lb = cent[0] - dim/2 , cent[1] - dim/2, cent[2] - dim/2 
            ub = cent[0] + dim/2, cent[1] + dim/2, cent[2] + dim/2
            lb = [int(lb[0]),int(lb[1]),int(lb[2])]
            ub = [int(ub[0]),int(ub[1]),int(ub[2])]
            if lb[0]<=0 or lb[1]<=0 or lb[2]<=0 or ub[0]>=volume_numpy.shape[0] or ub[1]>=volume_numpy.shape[1] or ub[2]>=volume_numpy.shape[2]:
                    continue
                
            try:
                centroid,size_d,thetas = location_vector_utils.extract_position_vector(numpy_labels, int(lab))
                loc_vector = np.concatenate((centroid, size_d, thetas))
            except IndexError as rw:
                print(f"Caught a runtime warning: {rw}")
                continue

            cl = [cur_class.values[0]]
            
            vol_to_add = volume_numpy[lb[0]:ub[0],lb[1]:ub[1],lb[2]:ub[2]].astype(np.float32)

            img_list.append(vol_to_add)
            y_list.append(np.array(cl))
            list_loc_vector.append(loc_vector.astype(np.float32))
        
        for i in range(len(img_list)):
            temp = img_list[i]
            temp = torch.from_numpy(temp) 
            img_list[i]=temp[np.newaxis,...]
            temp2 = list_loc_vector[i]
            temp2 = torch.from_numpy(temp2) 
            list_loc_vector[i]=temp2[np.newaxis,...]

        img_list = tuple(img_list) 

        return img_list,y_list,list_loc_vector
    
    def __len__(self):
        return self.n_samples
