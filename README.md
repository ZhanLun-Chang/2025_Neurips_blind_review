- [Fast Adaptation Federated Learning Framework](#fast-adaptation-federated-learning-framework)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. Create and activate the Conda environment](#2-create-and-activate-the-conda-environment)
    - [3. Install PyTorch with the appropriate CUDA version](#3-install-pytorch-with-the-appropriate-cuda-version)
    - [4. Install remaining Python packages](#4-install-remaining-python-packages)
  - [Usage](#usage)
    - [Example: FedAvg on CIFAR-10](#example-fedavg-on-cifar-10)
  - [Contributing](#contributing)
  - [License](#license)


# Fast Adaptation Federated Learning Framework

A PyTorch-based framework for evaluating federated learning algorithms under dynamic client and data distributions. Supports both image and text classification tasks with customizable label distributions and a variety of federated learning methods.

## Repository Structure

- **`fast_adapt_dataloader.py`**: Data loading utilities for:
  - **Image datasets**: MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100, TinyImageNet  
  - **Text dataset**: AG News  
  - Tokenization, preprocessing, and caching for efficient batch loading

- **`fast_adpat_net.py`**: Model architectures including:
  - `net` factory function  
  - `CNN_SVHN`, `Res18Con`, `SVHN_Cons`, `MobileNetV3SmallFeatureExtractor`, `SVHNEfficient`  
  - `FastText` text classifier

- **`fast_adpat_utils.py`**: Federated learning utilities:
  - `Client` class supporting FedAvg, FedProx, SCAFFOLD, MOON, FedACG, continuous distillation  
  - Helper functions: `compute_client_weights`, `initial_model_construction`, `run_pilot_stage`, `polynomial_decay_lr_schedule`, `maybe_download`, etc.

- **`fast_adpat_parser.py`**: Command-line argument parsing:
  - Core parameters: `--dataset_name`, `--algorithm`, `--in_session_label_dist`, `--dirichlet_alpha`, `--seed`, `--gpu_index`  
  - Algorithm-specific hyperparameters: `--prox_alpha`, `--moon_mu`, `--moon_tau`, `--kl_coefficient`, `--acg_beta`, `--acg_lambda`, `--footprint_num_iteration`, `--similarity`, `--similarity_scale`, etc.

- **`fast_adpat_training.py`**: Main training script:
  1. Parses arguments and sets random seeds  
  2. Downloads or loads the specified dataset  
  3. Initializes global and local models  
  4. Runs pilot sessions (warm-up) and main training sessions  
  5. Logs training progress, evaluates accuracy, and saves results/plots

- **`clean_fast_adapt.yaml`**: Conda environment specification for reproducibility

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fast-adapt-fl.git
cd fast-adapt-fl
````

### 2. Create and activate the Conda environment

This repository includes a `clean_fast_adapt.yaml` file that specifies Python and core dependencies.

```bash
conda env create -f clean_fast_adapt.yaml
conda activate fast_adapt
```

> If you prefer a custom environment name, use:
>
> ```bash
> conda env create -n <env_name> -f clean_fast_adapt.yaml
> conda activate <env_name>
> ```

### 3. Install PyTorch with the appropriate CUDA version

Depending on your GPU and CUDA toolkit, select the matching PyTorch build. For CUDA 12.1, run:

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

> Make sure to choose the `+cuXX` variant that matches your hardware (e.g., `+cu118` for CUDA 11.8, `+cpu` for CPU-only).

### 4. Install remaining Python packages

```bash
pip install -r requirements.txt
```

or individually:

```bash
pip install torchvision transformers datasets scikit-learn pandas nltk matplotlib pillow requests tqdm
python -m nltk.downloader punkt
```

## Usage

Run the training script with required arguments:

```bash
python fast_adpat_training.py \
  --dataset_name ag_news \
  --algorithm dyn \
  --in_session_label_dist dirichlet \
  --dirichlet_alpha 0.5 \
  --seed 42 \
  --gpu_index 0 \
  --batch_size 32 \
  --prox_alpha 0.1 \
  --moon_mu 0.2 \
  --moon_tau 0.1 \
  --kl_coefficient 0.1 \
  --acg_beta 0.5 \
  --acg_lambda 0.5 \
  --footprint_num_iteration 10 \
  --similarity two_norm \
  --similarity_scale 0.1
```

### Example: FedAvg on CIFAR-10

```bash
python fast_adpat_training.py \
  --dataset_name cifar10 \
  --algorithm fedavg \
  --in_session_label_dist dirichlet \
  --dirichlet_alpha 0.3 \
  --seed 123 \
  --gpu_index 0 \
  --batch_size 64 \
  --footprint_num_iteration 5 \
  --similarity two_norm \
  --similarity_scale 1.0
```


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for new datasets, algorithms, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


