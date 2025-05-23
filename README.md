
# Federated Learning with Dynamic Client Arrival and Departure: Convergence and Rapid Adaptation via Initial Model Construction

This repository contains the official PyTorch-based codebase for our paper under review at **NeurIPS 2025**.

## Repository Structure

- **`fast_adapt_dataloader.py`**: Data loading utilities for:
  - **Image datasets**: MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100, TinyImageNet  
  - **Text dataset**: AG News  
  - Includes tokenization, preprocessing, and caching for efficient batch loading

- **`fast_adpat_net.py`**: Model architectures, featuring:
  - `net` for MNIST & Fashion-MNIST 
  - `SVHNEfficient` for SVHN 
  - `ResNetFeatureExtractor` (ResNet-18 for CIFAR-10/100, ResNet-34 for TinyImageNet)
  - `FastText` for AG News

- **`fast_adpat_utils.py`**: Federated learning utilities:
  - `Client` class supporting multiple algorithms: FedAvg, FedProx, SCAFFOLD, MOON, FedACG, and continuous distillation  
  - Helper functions:  
    `compute_client_weights`, `initial_model_construction`, `run_pilot_stage`,  
    `polynomial_decay_lr_schedule`, `maybe_download`, and others

- **`fast_adpat_parser.py`**: Command-line argument parsing:
  - Core options: `--dataset_name`, `--algorithm`, `--in_session_label_dist`, `--dirichlet_alpha`, `--seed`, `--gpu_index`  
  - Algorithm-specific hyperparameters:  
    `--prox_alpha`, `--moon_mu`, `--moon_tau`, `--kl_coefficient`, `--acg_beta`, `--acg_lambda`,  
    `--footprint_num_iteration`, `--similarity`, `--similarity_scale`, etc.

- **`fast_adpat_training.py`**: Main training script:
  1. Parses arguments and sets random seeds  
  2. Downloads or loads the specified dataset  
  3. Initializes global and local models  
  4. Runs pilot sessions (warm-up) and main training sessions  
  5. Logs progress, evaluates accuracy, and saves results/plots

- **`clean_fast_adapt.yaml`**: Conda environment specification for reproducibility

- **`lr_poly.json`**: Predefined learning rate configurations for the polynomial decay schedule for each dataset–algorithm pair. Example:
  ```json
  {
    "mnist": {
      "avg": [0.001, 0.6],
      "prox": [0.001, 0.6],
      "acg": [0.001, 0.6],
      "moon": [0.001, 0.6],
      "scaffold": [0.03, 0.9]
    },
    "fmnist": { /* … */ },
    "SVHN":   { /* … */ },
    "ag_news":{ /* … */ },
    "cifar10":{ /* … */ },
    "cifar100":{ /* … */ },
    "tinyimagenet":{ /* … */ }
  }

### Polynomial Decay Learning Rate Schedule

The function `calculate_polynomial_decay_lr_schedule` in `fast_adpat_utils.py` implements the polynomial decay learning rate schedule:

```python
def calculate_polynomial_decay_lr_schedule(
    initial_client_lr: float,
    total_communication_rounds: int,
    power: float = 1.0,
    min_client_lr: float = 0.0
) -> List[float]:
    """
    Calculates a polynomial decay learning rate schedule:
        lr(t) = (initial_client_lr - min_client_lr) * (1 - t / total_communication_rounds)^power
                + min_client_lr

    Args:
        initial_client_lr: The starting (peak) learning rate.
        total_communication_rounds: Total number of rounds over which to decay.
        power: Exponent for the decay curve (p=1 → linear, p>1 → steeper tail, p<1 → gentler tail).
        min_client_lr: The floor learning rate at the end of the schedule.

    Returns:
        A list of length `total_communication_rounds` where
        each element is the LR for that round (rounds indexed from 0 to total_communication_rounds-1).
    """
    lr_values: List[float] = []
    delta = initial_client_lr - min_client_lr

    for t in range(total_communication_rounds):
        fraction = 1.0 - (t / total_communication_rounds)
        lr_t = delta * (fraction ** power) + min_client_lr
        lr_values.append(lr_t)

    return lr_values
```

In all use cases, the decay `power` is fixed at **0.9**. The following example demonstrates how to configure the schedule for the **MNIST** dataset using the **FedAvg** algorithm, with values sourced from the `lr_poly.json` file.

This polynomial decay learning rate schedule is applied **per session**. At the beginning of each session, the initial learning rate is set to `0.001`, and it gradually decays to `0.001 * 0.6` by the final communication round within the session.

> **Note**: The values `0.001` and `0.6` are specific to the **MNIST–FedAvg** pair. For other dataset–algorithm combinations, refer to the corresponding entries in the `lr_poly.json` file.

```python
lr_list = calculate_polynomial_decay_lr_schedule(
    initial_client_lr = 0.001,
    total_communication_rounds = args.num_rounds_actual,
    power = 0.9,
    min_client_lr = 0.001 * 0.6
)
```

## Installation

### 1. Create and activate the Conda environment

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

### 2. Install PyTorch with the appropriate CUDA version

Depending on your GPU and CUDA toolkit, select the matching PyTorch build. For CUDA 12.1, run:

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

> Make sure to choose the `+cuXX` variant that matches your hardware (e.g., `+cu118` for CUDA 11.8, `+cpu` for CPU-only).


## Example Training Command

The following command runs the federated training script with a specific experimental setup:

- **Dataset**: `cifar10`  
- **Algorithm**: `scaffold` (SCAFFOLD FL method with variance reduction)  
- **Label Distribution**:  
  - In-session label distribution is Dirichlet with concentration parameter `α=0.7`  
  - No cross-session label overlap (`--cross_session_label_overlap 0.0`)  
- **Baselines Enabled**:  
  - Proposed initial baseline: `--initial 1`  
  - Average baseline: `--average 1`  
  - Previous baseline: `--previous 1`  
  - Continuous baseline is disabled (`--continuous 0`)  
- **Sessions and Rounds**:  
  - 7 total sessions (`--num_sessions 7`), including 1 pilot session (`--num_sessions_pilot 1`)  
  - 100 communication rounds each for both pilot and actual sessions  
- **Clients**: 100 clients per session (`--num_clients 100`)  
- **Learning Rate Configuration**:  
  - Uses polynomial decay schedule defined in `lr_poly.json`  
- **Training Configuration**:  
  - Local training: 5 SGD steps with batch size 128 (`--num_SGD_training 5`, `--batch_size_training 128`)  
  - Gradient calculation: 5 SGD steps with batch size 128 (`--num_SGD_grad_cal 5`, `--batch_size_grad_cal 128`)  
  - Optimizer momentum: 0.9  
- **Gradient Computation**:  
  - `--num_round_grad_cal 1` indicates that only **1 communication round** is used to compute the gradient  
- **Algorithm-Specific Parameters**:  
  All parameters (e.g., `prox_alpha`, `moon_mu`, `kl_coefficient`, `acg_beta`, etc.) are included to enable switching between different algorithms without modifying the code.
- **GPU and Reproducibility**:  
  - Runs on GPU index 0 (`--gpu_index 0`)  
  - Uses fixed seed `300` to ensure reproducibility (`--seed 300`). This controls randomness in dataset splits, weight initialization, and other stochastic processes.

> **Note**: The following arguments correspond to key variables in our algorithm pseudocode:  
> - `--num_round_grad_cal` → **V** (rounds for gradient computation)  
> - `--num_sessions` → **S** (total number of sessions)  
> - `--num_sessions_pilot` → **P** (number of pilot sessions)  
> - `--similarity_scale` → **R** (similarity scale)  
> - `--num_rounds_pilot` and `--num_rounds_actual` → **T** (number of communication rounds per session)

```bash
# Run the Python script
python -u fasy_adapt_training.py \
    --dataset_name 'cifar10' \
    --algorithm 'scaffold' \
    --cross_session_label_overlap 0.0 \
    --in_session_label_dist "dirichlet" \
    --dirichlet_alpha 0.7 \
    --seed 300\
    --initial 1\
    --average 1\
    --previous 1\
    --continuous 0\
    --num_clients 100\
    --num_sessions 7\
    --num_sessions_pilot 1 \
    --num_rounds_pilot 100 \
    --num_rounds_actual 100\
    --lr_config_path 'lr_poly.json' \
    --momentum 0.9 \
    --gpu_index 0 \
    --num_SGD_training 5 \
    --batch_size_training 128 \
    --num_SGD_grad_cal 5 \
    --batch_size_grad_cal 128 \
    --prox_alpha 1.0 \
    --moon_mu 1.0 \
    --moon_tau 1.0 \
    --kl_coefficient 0.5\
    --acg_beta 0.1\
    --acg_lambda 0.5\
    --num_round_grad_cal 1 \
    --similarity "two_norm" \
    --similarity_scale 10.0
```

## Saved Outputs

### Run Identifier and Output Artifacts

Each execution is uniquely tagged with a `run_identifier`:

- When running under **SLURM**, it is set to:  
  `job_<SLURM_JOB_ID>`
- Otherwise, it defaults to:  
  `local_ET_<ISO-timestamp>_pid_<process_id>`  
  using the local time in the Eastern Time zone.

After training, the following artifacts are generated (all paths are relative to the current working directory):

- **Plots**:  
  `training_plot/<dataset_name>/<run_identifier>_<key>.png`  
  where `<key>` represents each baseline (`initial`, `average`, `continuous`, `previous`).

- **Training Values**:  
  Raw metric series stored as JSON files in:  
  `training_values/<dataset_name>/<run_identifier>/<key>.json`

- **Arguments**:  
  Full script arguments saved as JSON in:  
  `args_namespace/<dataset_name>/<run_identifier>.json`

- **Training Order**:  
  The client session sequence is saved in:  
  `training_order/<run_identifier>.json`

- **Similarity Checks**:  
  Original and normalized similarity values saved as:  
  `<similarity>_value/<dataset_name>/<run_identifier>/{original.json, normalized.json}`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


