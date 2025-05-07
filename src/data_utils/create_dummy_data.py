import numpy as np
import nibabel as nib
import pandas as pd
import random
from scipy.ndimage import binary_dilation, binary_fill_holes
import os

def create_3d_segmentation_mask(shape, n_objects, max_radius=5):
    """
    Creates a 3D numpy array with `n_objects` non-overlapping labeled regions.
    Each region will be labeled with a unique integer from 1 to n.
    
    Parameters:
        shape (tuple): Shape of the 3D array (e.g., (64, 64, 64)).
        n_objects (int): Number of distinct regions to create.
        max_radius (int): Maximum radius of the random objects.

    Returns:
        np.ndarray: 3D segmentation mask with labeled regions.
    """
    mask = np.zeros(shape, dtype=np.int32)
    
    for label in range(1, n_objects + 1):
        success = False
        attempts = 0
        
        while not success and attempts < 100:
            attempts += 1
            # Random center location for the object
            center = tuple(np.random.randint(0, s) for s in shape)
            
            # Create a binary mask for the current object
            object_mask = np.zeros(shape, dtype=bool)
            object_mask[center] = 1
            for _ in range(random.randint(1, max_radius)):
                object_mask = binary_dilation(object_mask)
            object_mask = binary_fill_holes(object_mask)
            
            # Check if this region overlaps with existing regions
            if np.any(mask[object_mask]):
                continue
            
            # Add the non-overlapping object to the main mask
            mask[object_mask] = label
            success = True
    
    return mask

MAIN_PATH = "../data/VOLS"
# create dir and subdirs if they don't exist
os.makedirs(MAIN_PATH, exist_ok=True)

def save_nii(mask, filename):
    """
    Saves a 3D numpy array as a .nii.gz file.
    
    Parameters:
        mask (np.ndarray): 3D segmentation mask.
        filename (str): Output filename for the .nii.gz file.
    """
    img = nib.Nifti1Image(mask, affine=np.eye(4))
    nib.save(img, os.path.join(MAIN_PATH,filename))

def collect_patient_data(n_objects, filename):
    """
    Collects data for a single patient with columns: Patient, Label, Class.
    
    Parameters:
        n_objects (int): Number of objects to list in the data.
        filename (str): Patient filename to associate with each row.

    Returns:
        pd.DataFrame: Data for the patient with columns `Patient`, `Label`, and `Class`.
    """
    data = {
        "Patient": [filename] * n_objects,
        "Label": list(range(1, n_objects + 1)),
        "Class": [random.randint(0, 1) for _ in range(n_objects)]
    }
    return pd.DataFrame(data)

def create_patient_files(num_patients=4, shape=(64, 64, 64), n_objects=80):
    """
    Creates segmentation masks and a single CSV file for multiple patients.
    
    Parameters:
        num_patients (int): Number of patients/files to create.
        shape (tuple): Shape of the 3D mask.
        n_objects (int): Number of distinct regions per mask.
    """
    all_patient_data = []

    for i in range(1, num_patients + 1):
        # Define the filename for the patient
        filename = f"patient_{i:03d}"
        
        # Generate segmentation mask
        mask = create_3d_segmentation_mask(shape, n_objects)
        
        # Save the mask as a .nii.gz file
        nii_filename = f"{filename}.nii.gz"
        save_nii(mask, nii_filename)
        
        # Collect individual patient data
        patient_data = collect_patient_data(n_objects, filename)
        all_patient_data.append(patient_data)
    
    # Concatenate all patient data into a single DataFrame and save it
    combined_df = pd.concat(all_patient_data, ignore_index=True)
    combined_df.to_csv("patients_labels.csv", index=False)

# Run the function to create files for 3 patients
create_patient_files(num_patients=4,n_objects=80)

print("3D segmentation masks and single CSV file created successfully.")


def create_file_paths_csv(folder_path, output_csv="dataset.csv"):
    """
    Reads all .nii.gz files from a specified folder and creates a CSV file 
    with the full paths of each file.
    
    Parameters:
        folder_path (str): Path to the folder containing the .nii.gz files.
        output_csv (str): Name of the output CSV file to save file paths.
    """
    # List to hold file paths
    file_paths = []
    
    # Walk through the folder and find all .nii.gz files
    for root, _, files in os.walk(folder_path):
        print(files)
        for file in files:
            if file.endswith(".nii.gz"):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    
    # Create a DataFrame from the file paths
    df = pd.DataFrame({"FilePath": file_paths})
    
    # Save to CSV
    df.to_csv(output_csv, index=False,header=False)

# Usage example
# Set your folder path where .nii.gz files are stored
#folder_path = "/home/tpv/Representation-Learning-Sem-Pos/data_simulate/SEG"  # replace with your folder path
create_file_paths_csv(MAIN_PATH, "dataset_simulate.csv")

#COPY THE FOLDER ../data/VOLS to ../data/SEGS_MULTI
import shutil
src_dir = os.path.abspath(os.path.join('..', 'data', 'VOLS'))
dst_dir = os.path.abspath(os.path.join('..', 'data', 'SEGS_MULTI'))
shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

print("CSV file with file paths created successfully.")