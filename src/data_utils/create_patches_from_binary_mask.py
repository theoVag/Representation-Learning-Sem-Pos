import os
import random
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import label, find_objects


def process_binary_masks_and_sample(vols_folder="../data/VOLS",
                                    segs_folder="../data/SEGS",
                                    output_folder="../data/SEGS_MULTI",
                                    class0_pct=0.1):
    """
    Reads binary masks (class 1) and corresponding images,
    labels each connected component as a separate region (class 1),
    samples background patches (class 0) matching region sizes,
    and saves multi-label masks plus a CSV of regions.

    Params:
      vols_folder: folder with NIfTI images (optional use)
      segs_folder: folder with binary segmentation NIfTIs
      output_folder: where to save new multi-labeled masks and CSV
      class0_pct: fraction of class1 count to sample as class0
    """
    os.makedirs(output_folder, exist_ok=True)
    rows = []

    for fname in sorted(os.listdir(segs_folder)):
        if not fname.endswith('.nii.gz'):
            continue
        patient = os.path.splitext(os.path.splitext(fname)[0])[0]

        # load binary mask (class1)
        seg_img = nib.load(os.path.join(segs_folder, fname))
        bin_data = seg_img.get_fdata() > 0

        # label connected components => multi-label class1 mask
        class1_mask, n_class1 = label(bin_data)

        # collect shapes via bounding boxes
        objects = find_objects(class1_mask)
        shapes = []  # (dz, dy, dx)
        for sl in objects[1:]:  # skip background index 0
            if sl is None:
                shapes.append((1,1,1))
            else:
                dz = sl[0].stop - sl[0].start
                dy = sl[1].stop - sl[1].start
                dx = sl[2].stop - sl[2].start
                shapes.append((dz, dy, dx))

        # determine number of class0 patches
        num_class0 = int(class0_pct * n_class1)

        # sample background patches
        dims = bin_data.shape
        occupied = bin_data.copy()
        class0_patches = []
        tries = 0
        while len(class0_patches) < num_class0 and tries < num_class0 * 100:
            tries += 1
            # pick random shape
            dz, dy, dx = random.choice(shapes)
            # random origin within bounds
            z0 = random.randint(0, dims[0] - dz)
            y0 = random.randint(0, dims[1] - dy)
            x0 = random.randint(0, dims[2] - dx)
            patch = np.zeros(dims, bool)
            patch[z0:z0+dz, y0:y0+dy, x0:x0+dx] = True
            if np.any(occupied & patch):
                continue
            class0_patches.append(patch)
            occupied |= patch

        # build combined mask
        new_mask = np.zeros_like(class1_mask, dtype=np.int32)
        label_id = 1
        # add class1 regions
        for lbl in range(1, n_class1+1):
            region = (class1_mask == lbl)
            new_mask[region] = label_id
            rows.append({'Patient': patient, 'Label': label_id, 'Class': 1})
            label_id += 1
        # add class0 regions
        for patch in class0_patches:
            new_mask[patch] = label_id
            rows.append({'Patient': patient, 'Label': label_id, 'Class': 0})
            label_id += 1

        # save new multi-label mask
        out_img = nib.Nifti1Image(new_mask,
                                  affine=seg_img.affine,
                                  header=seg_img.header)
        nib.save(out_img, os.path.join(output_folder, fname))

    # save CSV
    df = pd.DataFrame(rows)
    df.to_csv('patients_labels.csv',
              index=False)

    print(f"Processed {len(rows)} regions across {len(os.listdir(segs_folder))} patients.")


if __name__ == '__main__':
    process_binary_masks_and_sample()
