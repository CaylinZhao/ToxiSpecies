# Meta_Tox: Meta-Learning for Multi-Species Molecular Toxicity Prediction

Meta_Tox is a deep learning framework based on **Meta-Learning** designed for predicting molecular toxicity across diverse species (e.g., from rats to humans) and different experimental endpoints. It addresses the challenge of data heterogeneity and scarcity by utilizing **Feature Adapter (FA)** and **Label Adapter (LA)** mechanisms, enabling accurate toxicity prediction with limited samples (Few-shot learning). This codebase supports the reproducibility of results presented in our manuscript submitted to **JCIM**.

## 📁 Repository Structure

| File | Type | Description |
| :--- | :--- | :--- |
| **Main_FA.py** | Main Script | Entrance for training and evaluating models using the **Feature Adapter** mechanism. |
| **Main_LA.py** | Main Script | Entrance for training and evaluating models using the **Label Adapter** mechanism. |
| **Model.py** | Architecture | PyTorch implementation of the base MLP network and the FA/LA adaptation layers. |
| **Train.py** | Core Logic | Implementation of the Meta-Learning inner-loop (task adaptation) and outer-loop (meta-update). |
| **Task_split.py** | Preprocessing | Splits raw toxicity data into tasks according to different experimental settings. |
| **Sampler.py** | Data Loader | Samples Support Sets and Query Sets for meta-learning tasks. |
| **Ensemble_DA.py** | Integration | Combines FA and LA results (Dual Adapter) for ensemble evaluation. |
| **GHS_cl.py** | Post-processing | Converts regression values (e.g., LC50/LD50) into GHS Toxicity Categories (Levels 1-5). |
| **early_stopping.py** | Utility | Monitors validation performance to prevent overfitting. |
| **data_stat.csv** | Statistics | Statistical info for each species/task (sample size, mean, variance, etc.). |

## 🧪 Reproducing Manuscript Results

To reproduce the results presented in the manuscript, follow these steps sequentially:

### 1. Environment Requirements
This project is tested with **Python 3.8+**. It is recommended to use the following environment configuration:

#### Core Dependencies (Fixed Versions):
- **PyTorch**: `2.2.1+cu121`
- **RDKit**: `2023.09.6` (for preprocessing SMILES to fingerprints)
- **Pandas**: `2.3.1`
- **NumPy**: `1.26.4`
- **Scikit-learn**: `1.7.1`
- **SciPy**: For Pearson/Spearman correlation analysis.
- **Matplotlib**: For training curve visualization.

#### Setup via Conda:
```bash
conda create -n metatox python=3.8
conda activate metatox
conda install -c rdkit rdkit=2023.09.6
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas==2.3.1 numpy==1.26.4 scikit-learn==1.7.1 scipy matplotlib
```

### 2. Data Preparation
- **Input Files**: The required input files are molecular SMILES and 2048-bit Morgan Fingerprints, stored in `Data.zip`.
- **Pre-processing**:
  ```bash
  unzip Data.zip
  python Task_split.py --setting 1_1  # Scenario 1: Random task split
  # OR
  python Task_split.py --setting 2_1  # Scenario 2: Cross-species extrapolation
  ```

### 3. Model Training (Few-Shot Adaptation)
The manuscript results are based on comparing and integrating FA and LA pathways:
- **Feature Adapter (FA) Training**:
  ```bash
  python Main_FA.py --input_dim 2048 --k_shot_train 10 --n_q_train 10 --update_step_inner 5
  ```
- **Label Adapter (LA) Training**:
  ```bash
  python Main_LA.py --input_dim 2048 --k_shot_train 10 --n_q_train 10 --update_step_inner 5
  ```

### 4. Integration and GHS Classification
To obtain the final integrated metrics and GHS classification levels:
```bash
python Ensemble_DA.py
python GHS_cl.py
```

## 📊 Expected Outputs
- **Log Files**: Training logs with loss reduction and validation metrics are printed to the console.
- **Model Weights**: `.pth` files saved in the `Models/` directory.
- **Evaluation Tables**: CSV files in the `Results/` directory containing R2, RMSE, and correlation coefficients for each test task.
- **Toxicity Categories**: GHS classification results outputted as hazard levels (1-5).

## 🛠️ Configuration & Hyperparameters
- `--input_dim`: Dimension of molecular fingerprints (default: `2048`).
- `--k_shot_train`: Number of support samples per task (Few-shot setting).
- `--update_step_inner`: Number of gradient steps for task-specific adaptation.
- `--lr_inner` / `--lr_outer`: Learning rates for inner and outer loops.

## 🧩 Key Workflows & Code Logic
- **Meta-Learning Loop**: Located in `Train.py`. The `meta_train` function manages the cross-task optimization process.
- **Task Adaptation**: `Model.py` contains `FeatureAdapter` and `LabelAdapter`, which allow the model to adjust for specific species data distributions.
