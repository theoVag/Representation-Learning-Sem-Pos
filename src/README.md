# Project Title

**“T. P. Vagenas, M. Vakalopoulou, C. Sachpekidis, A. Dimitrakopoulou-Strauss and G. K. Matsopoulos, "Representation learning in PET scans enhanced by semantic and 3D position specific characteristics," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2025.3566996.”**  
🔗 [Link to paper](https://ieeexplore.ieee.org/document/10985918)

---

## 🗂️ Repository Structure

```
├── src/
│   ├── data_utils/            # Data preparation & dummy data scripts
│   ├── configs/               # Configuration files
│   ├── train_representation_learning.py
│   ├── train_classification_pos.py
│   ├── test_classification_pos.py
│   └── ...
├── environment.yml            # Conda environment spec
├── requirements.txt           # pip dependencies
└── README.md                  # This file
```

---

## ⚙️ Requirements

- **Languages & Frameworks**:  
  - Python 3.8+  
  - PyTorch 1.13.0  
  - SimpleITK 2.2.1  
  - Pylightly  
  - MONAI 1.0.1  
  - wandb 0.15.3  
  - pytorch-metric-learning  


---

## 🐍 Creating the Environment

We provide Conda-based setup via `environment.yml`.

```bash
conda env create -f environment.yml
conda activate your_env_name
```

## 📦 Data Preparation & Simulation

All data‐prep and dummy‐data scripts live under `src/data_utils/`.

```bash
cd src
```

### Execution of Simulation with Dummy Data

This script generates dummy patient data in `.nii.gz` and accompanying CSVs:

```bash
python data_utils/create_dummy_data.py
```

- **Required outputs**  
  - A CSV listing patient NIfTI file paths  
  - A CSV with columns: `Patient,Label,Class` (for multi‐label data with regions)  
- **Folder structure**  
  ```
  data/
  ├── VOLS/
  ├── SEGS/
  └── SEGS_MULTI/
  ```


### Train/Validation/Test Split

Adjust the `MAIN_PATH` variable in your config files (`configs/`) to point at the appropriate CSVs to control dataset splits.

---

## 🚀 Running the Main Models

From the `src/` directory:

### 1. Representation Learning with Positional Encoding

```bash
python train_representation_learning.py     --model_name resnet18     --config configs/config.py
```

### 3. Classification with Positional Encoding

```bash
python train_classification_pos.py     --backbone_name resnet18     --config configs/config_classification.py
```

```bash
python test_classification_pos.py     --backbone_name resnet18     --config configs/config_classification.py
```

---


### Generating Multi‐Label Regions from Binary Masks

If you have only binary masks (in `data/SEGS`), you can split them into labeled regions:

```bash
python data_utils/process_binary_masks_and_sample.py     "../data/SEGS" "../data/SEGS_MULTI"
```

- The resulting NIfTI masks in `SEGS_MULTI` will have unique integer IDs per region (e.g., 1–130 for 130 ROIs).

## 📑 Citation

If you use this code, please cite our paper:

```bibtex
@ARTICLE{10985918,

  author={Vagenas, Theodoros P. and Vakalopoulou, Maria and Sachpekidis, Christos and Dimitrakopoulou-Strauss, Antonia and Matsopoulos, George K.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Representation learning in PET scans enhanced by semantic and 3D position specific characteristics}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2025.3566996}}

```

