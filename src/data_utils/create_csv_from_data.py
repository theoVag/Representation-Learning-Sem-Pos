import os
import random
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import label

def process_segmentations(input_folder="../data/SEGS",
                          output_folder="../data/SEGS_MULTI"):
    # ensure output dir exists
    os.makedirs(output_folder, exist_ok=True)

    all_rows = []
    # loop over all nii.gz in the source folder
    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith(".nii.gz"):
            continue

        in_path  = os.path.join(input_folder,  fname)
        out_path = os.path.join(output_folder, fname)

        # load binary mask, threshold to boolean
        img  = nib.load(in_path)
        data = img.get_fdata() > 0

        # connected components: returns (labeled_array, n_labels)
        labeled, n_labels = label(data)

        # save the new multi-label mask
        new_img = nib.Nifti1Image(labeled.astype(np.int32),
                                  affine=img.affine,
                                  header=img.header)
        nib.save(new_img, out_path)

        # collect per-label info for CSV
        patient_id = os.path.splitext(os.path.splitext(fname)[0])[0]
        for lbl in range(1, n_labels + 1):
            all_rows.append({
                "Patient": patient_id,
                "Label":   lbl,
                "Class":   random.randint(0, 1)  # replace it
            })

    # write combined patients_labels.csv
    df = pd.DataFrame(all_rows)
    df.to_csv("patients_labels.csv",
              index=False)

    # write dataset.csv listing all output .nii.gz paths (no header)
    out_files = [os.path.join(output_folder, f)
                 for f in sorted(os.listdir(output_folder))
                 if f.endswith(".nii.gz")]
    pd.DataFrame({"FilePath": out_files}) \
      .to_csv("dataset_simulate.csv",
              index=False, header=False)

    print(f"Processed {len(out_files)} files, "
          f"{len(df)} labels saved.") 

# Example usage:
if __name__ == "__main__":
    process_segmentations()
