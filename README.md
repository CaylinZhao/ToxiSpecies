# ToxiSpecies

ToxiSpecies is a task-aware meta-learning framework for toxicity prediction across multiple assay settings. It is designed to learn from small, heterogeneous toxicology tasks by combining a shared molecular predictor with two complementary adapters:

- Feature Adapter (FA)
- Label Adapter (LA)

This GitHub upload contains the benchmark data and the core code for sampling, training, and testing. It does not include manuscript figures, result artifacts, or model checkpoints.

## Overview

The project focuses on predicting toxicological endpoints across multiple species, exposure routes, and assay settings. The uploaded repository mainly provides:

- the benchmark data under `Data/`
- sampling and task-splitting utilities
- training and testing code for the ToxiSpecies models
- shared model and argument definitions used by the pipeline

## Model Input and Output

The core models operate on task-wise molecular feature batches.

- Input features: 2048-dimensional molecular fingerprints or feature vectors
- Input labels: scalar toxicity values for each compound in a task
- Training input structure: support sets and query sets sampled per task
- Model output: a single predicted toxicity value for each input compound
- Evaluation output: regression metrics such as MAE, RMSE, $R^2$, Pearson correlation, and Spearman correlation

In the label-adaptation setting, the model also performs a task-specific label transformation before mapping predictions back to the original toxicity scale.

## Repository Layout

- `Data/3.Task split/`: task-level data split used for the benchmark
- `Sampler.py`: task and episode sampling utilities
- `Train.py`: training and evaluation workflow
- `Main_FA.py`, `Main_LA.py`, `Ensemble_DA.py`: entry points for the core ToxiSpecies models
- `Model.py`: model components, including the adapters
- `Task_split.py`, `Data_split.py`: data and task split helpers
- `args.py`: command-line arguments and default hyperparameters
- `run_ToxiSpecies.sh`: launches the main training pipeline

## Requirements

The codebase is written in Python and uses PyTorch-based training. The repository was verified in the conda environment with the following key versions:

- PyTorch 2.2.1+cu121
- RDKit 2023.09.6

A working environment should also include at least:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- scikit-learn
- SciPy
- matplotlib
- seaborn
- RDKit

RDKit is required for the chemistry-related preprocessing used in the repository.

## Installation

The repository does not bundle a lock file, so install the dependencies in your preferred Python environment. The setup below matches the verified conda `base` environment:

```bash
conda activate base
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn scipy matplotlib seaborn
conda install -c conda-forge rdkit=2023.09.6 -y
```

If your environment already contains these versions, you can reuse it directly.

## Data

The benchmark data are organized under `Data/3.Task split/`. The task-level benchmark statistics are summarized in Table S1 of the manuscript.

Important note:

- the repository expects the task split files to be present under `Data/3.Task split/`
- most scripts read and write relative to the project root

## Quick Start

### 1. Activate your environment

```bash
conda activate base
```

If your local workflow uses RDKit-dependent preprocessing, make sure RDKit is available in the active environment.

### 2. Train the main ToxiSpecies models

```bash
bash run_ToxiSpecies.sh
```

This script launches the FA, LA, and DA pipelines across the predefined settings and learning rates.

### 3. Run testing or evaluation

The main scripts also provide task-level testing and evaluation routines. Typical outputs are written to local result directories when you run the training or testing code from the project root.

## Training and Evaluation Notes

- The training scripts use task-level meta-learning with support/query adaptation.
- The project uses multiple settings and repeated support-set sampling for evaluation; the exact protocol is controlled by the corresponding training scripts.
- The uploaded repository is intended for code and data release, so generated artifacts are expected to be created locally when you run the scripts.

## Reproducibility Tips

- Run all scripts from the project root so that relative paths resolve correctly.
- Keep the data split files unchanged when reproducing the published results.
- If you rerun the full pipeline, keep your local output directories separate from the uploaded source tree if you want to avoid mixing generated files with the release contents.

## Citation

If you use this repository, please cite the corresponding ToxiSpecies manuscript.
