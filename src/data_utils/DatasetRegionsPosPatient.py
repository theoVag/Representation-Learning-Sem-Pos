
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from glob import glob
import os,sys
import SimpleITK as sitk
import pandas as pd
import random
import time
import location_vector_utils

sys.path.insert(0,'../')

def find_centroid(label, labeled_array):
    indices = np.array(np.where(labeled_array == label))
    centroid = np.mean(indices, axis=1).astype(int)
    return centroid


class DatasetRegionsPosPatient(Dataset):
    
    def __init__(self,MAIN_PATH,
                 spatial_size=[32,32,32],
                 PATH_LABELS='patient_data_fixed_voxel_number.csv'):
        self.MAIN_PATH=MAIN_PATH
        self.spatial_size=spatial_size
        self.PATH_LABELS=PATH_LABELS
        #random.seed(1234)
        
        if MAIN_PATH.endswith(".csv"):
            fnames = pd.read_csv(MAIN_PATH,header=None)
            self.volumes_paths = fnames.iloc[:,0].tolist()
            self.segs_paths = [word.replace('VOLS','SEGS_MULTI') for word in self.volumes_paths]
            
            self.labels_csv = pd.read_csv(PATH_LABELS)

            patients_to_keep = [word.split('/')[-1].replace('.nii.gz','') for word in self.volumes_paths]

            self.labels_csv = self.labels_csv.loc[self.labels_csv['Patient'].isin(patients_to_keep),:]
            print(self.labels_csv)
            print(patients_to_keep)
            
            """sample_p1 = 0.1
            sample_p2 = 0.3 # htan 0.6
            if 'val' in MAIN_PATH and 'private' not in MAIN_PATH:
                sample_p1 = 0.1#0.2
                sample_p2 = 0.3#0.6 htan

            if 'private' in MAIN_PATH:
                sample_p1 = 0.1 #prepei polu mikro
                sample_p2 = 0.6 # htan 0.6
            
            if 'val' in MAIN_PATH and 'private' in MAIN_PATH:
                sample_p1 = 0.3
                sample_p2 = 0.6

            samples_p1 = self.labels_csv.loc[self.labels_csv["VoxelNumber"]<85,] # 64x12 83 htan
            samples_p1_cl0 = samples_p1.loc[samples_p1["Class"]==0,]
            samples_p1_cl1 = samples_p1.loc[samples_p1["Class"]==1,]
            
            samples_p1_cl0=samples_p1_cl0.sample(frac=sample_p1, replace=False) #0.05 # to kalo 0.1          
            samples_p2 = self.labels_csv.loc[self.labels_csv["VoxelNumber"]>=85,]
            samples_p2_cl0 = samples_p2.loc[samples_p2["Class"]==0,]
            samples_p2_cl1 = samples_p2.loc[samples_p2["Class"]==1,]
            
            samples_p2_cl0=samples_p2_cl0.sample(frac=sample_p2, replace=False) #0.1
           
            self.labels_csv = pd.concat([samples_p1_cl0,samples_p1_cl1,samples_p2_cl0,samples_p2_cl1],axis=0)"""
                        
            self.examples_per_iter = self.labels_csv.Patient.unique().tolist()
            
            self.n_samples = len(self.examples_per_iter)
    
    def __getitem__(self,index):
        
        dim=self.spatial_size[0]
        patient = self.examples_per_iter[index]
        cur_patient = self.labels_csv.loc[self.labels_csv['Patient']==self.examples_per_iter[index],:]


        img_list_tumor=[]
        y_list_tumor=[]
        img_list_notumor=[]
        y_list_notumor=[]
        list_loc_vector_tumor=[]
        list_loc_vector_notumor=[]
        start_timeout=time.time()

        path_vol = [word for word in self.volumes_paths if patient in word.split('/')[-1]][0]
        path_seg = [word for word in self.segs_paths if patient in word.split('/')[-1]][0]
        
        print(path_vol)
        volume = sitk.ReadImage(path_vol)
        labels = sitk.ReadImage(path_seg)
        volume = sitk.Cast(volume, sitk.sitkFloat32)
        
        # example for the private dataset (prepare images of the external dataset- Modify as needed)
        if False: 
            size = volume.GetSize()
            region_of_interest = [0, 0, 200, size[0], size[1], size[2] - 200]
            volume = sitk.RegionOfInterest(volume, region_of_interest[3:], region_of_interest[:3])
            labels = sitk.RegionOfInterest(labels, region_of_interest[3:], region_of_interest[:3])
            original_spacing = volume.GetSpacing()
            original_origin = volume.GetOrigin()
            volume = location_vector_utils.resample_image(volume, (original_spacing[0], original_spacing[1], 3),[1, 0, 0, 0, -1, 0, 0, 0, 1],original_origin, sitk.sitkLinear) #[original_origin[0],200,original_origin[2]]
            volume = sitk.Flip(volume,[False,True,False])
            labels = location_vector_utils.resample_image(labels, (original_spacing[0], original_spacing[1], 3),[1, 0, 0, 0, -1, 0, 0, 0, 1],original_origin, sitk.sitkNearestNeighbor)
            labels = sitk.Flip(labels,[False,True,False])


        labels=sitk.GetArrayFromImage(labels)

        #volume = sitk.IntensityWindowing(volume, 0, 40, 0, 1)
        volume = sitk.Normalize(volume)
        volume_numpy = sitk.GetArrayFromImage(volume)
        
        num_labels_1 = list(cur_patient.loc[:,'Label'])
        num_labels_2 = list(np.unique(labels))
        num_labels = list(set(num_labels_1) & set(num_labels_2))
        random.shuffle(num_labels)
        # for loop to extract the samples from each region with label lab
        for lab in num_labels:
            if lab==0: # do not use it for the background region: 0
                continue
            
            cur_class = cur_patient.loc[cur_patient['Label']==lab,'Class']
            cl = cur_class.values[0]
            
            # Extract the centroid of the region with label lab and finally the ROI for the sample
            cent_full=[find_centroid(lab,labels)]                
            for i_sample in range(len(cent_full)):
                cent = cent_full[i_sample]
                if cent[0]==-1:
                    continue
                lb = cent[0] - dim/2 , cent[1] - dim/2, cent[2] - dim/2 
                ub = cent[0] + dim/2, cent[1] + dim/2, cent[2] + dim/2
                lb = [int(lb[0]),int(lb[1]),int(lb[2])]
                ub = [int(ub[0]),int(ub[1]),int(ub[2])]

                if lb[0]<0 or lb[1]<0 or lb[2]<0 or ub[0]>=volume_numpy.shape[0] or ub[1]>=volume_numpy.shape[1] or ub[2]>=volume_numpy.shape[2]:
                    continue

                # Calculate the position vector
                try:
                    centroid,size_d,thetas = location_vector_utils.extract_position_vector(labels, int(lab))
                    loc_vector = np.concatenate((centroid, size_d, thetas))                    
                except RuntimeError as rw:
                    print(f"Caught a runtime error: {rw}")
                    continue
                
                # Save the image, class and position to the corresponding list for tumors and non-tumors
                vol_to_add = volume_numpy[lb[0]:ub[0],lb[1]:ub[1],lb[2]:ub[2]].astype(np.float32)
                if cl==1:
                    img_list_tumor.append(vol_to_add)
                    y_list_tumor.append(cl)
                    list_loc_vector_tumor.append(loc_vector.astype(np.float32))

                else:
                    img_list_notumor.append(vol_to_add)
                    y_list_notumor.append(cl)
                    list_loc_vector_notumor.append(loc_vector.astype(np.float32))

                # apply timeout if the process takes too long
                end_timeout=time.time()
                if (end_timeout-start_timeout)>800:
                    break
        
            
        volume=[]
        zipped = list(zip(img_list_tumor,list_loc_vector_tumor))
        try:
            samples1_vol,samples1_loc = zip(*zipped)
        except ValueError as ve:
            print(f"Error: {ve}")
            samples1_vol,samples1_loc = [], []
        
        zipped = list(zip(img_list_notumor,list_loc_vector_notumor))
        
        try:
            samples2_vol,samples2_loc = zip(*zipped)
        except ValueError as er:
            print("value error ", er)
            samples2_vol,samples2_loc = [],[]
        

        volume=[]
        for k in range(len(samples1_vol)):
            volume.append((samples1_vol[k],1,samples1_loc[k]))

        for k in range(len(samples2_vol)):
            volume.append((samples2_vol[k],0,samples2_loc[k]))

        return volume
    
    def __len__(self):
        return self.n_samples
