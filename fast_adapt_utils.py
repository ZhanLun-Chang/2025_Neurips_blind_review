import torch
from fast_adpat_net import net, ResNetFeatureExtractor, SVHNEfficient, FastText
import copy
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
import torchvision.models as tvmodels
import torch.nn.functional as F
import os
import torch.nn as nn
import traceback
from typing import Dict, List, Optional, Tuple, Callable, Any
from fast_adapt_dataloader import dict_to_tuple_collate
import logging
import torch.optim as optim
from pathlib import Path
from argparse import Namespace
from urllib.request import urlretrieve
import time


StateDict = Dict[str, torch.Tensor]

def _get_target_device(
    tensors: List[Optional[torch.Tensor]],
    explicit_device: Optional[torch.device] = None
) -> torch.device:
    """
    Determine the target device for state dict operations.

    Args:
        tensors: List of tensors from which to infer the device.
        explicit_device: If provided, this device will be used.

    Returns:
        The determined torch.device.
    """
    if explicit_device:
        return explicit_device
    for tensor in tensors:
        if tensor is not None:
            return tensor.device
    return torch.device("cpu")


def subtract_state_dicts(
    state_dict_a: Optional[StateDict],
    state_dict_b: Optional[StateDict],
    target_device: Optional[torch.device] = None
) -> Optional[StateDict]:
    """
    Subtracts state_dict_b from state_dict_a element-wise for keys common to both.

    Args:
        state_dict_a: The state dictionary to subtract from.
        state_dict_b: The state dictionary to subtract.
        target_device: Desired device for the resulting tensors. If None,
                       inferred from state_dict_a or defaults to CPU.

    Returns:
        A new state dictionary with differences, or None if inputs are invalid
        or no common keys exist. Warns if key sets differ.
    """
    if state_dict_a is None or state_dict_b is None:
        print("Warning: subtract_state_dicts received None input.")
        return None

    keys_a = set(state_dict_a.keys())
    keys_b = set(state_dict_b.keys())

    if keys_a != keys_b:
        print(f"Warning: Keys mismatch in subtract_state_dicts. "
              f"A-only: {keys_a - keys_b}, B-only: {keys_b - keys_a}")

    common_keys = keys_a & keys_b
    if not common_keys:
        print("Warning: No common keys found in subtract_state_dicts.")
        return None

    device = _get_target_device(list(state_dict_a.values()), target_device)
    result_dict: StateDict = {}

    for key in common_keys:
        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]
        if tensor_a is None or tensor_b is None:
            print(f"Warning: Found None tensor for key '{key}' during subtraction. Skipping key.")
            continue
        result_dict[key] = (
            tensor_a.to(device).clone().detach() -
            tensor_b.to(device).clone().detach()
        )

    return result_dict if result_dict else None


def add_state_dicts(
    state_dict_a: Optional[StateDict],
    state_dict_b: Optional[StateDict],
    target_device: Optional[torch.device] = None
) -> Optional[StateDict]:
    """
    Adds two state dictionaries element-wise, handling missing keys in either dict.

    Args:
        state_dict_a: The first state dictionary.
        state_dict_b: The second state dictionary.
        target_device: Desired device for the resulting tensors. If None,
                       inferred from inputs or defaults to CPU.

    Returns:
        A new state dictionary containing the sums. If one input is None, returns a
        copy of the other. Returns None if both inputs are invalid.
    """
    if state_dict_a is None and state_dict_b is None:
        print("Warning: add_state_dicts received two None inputs.")
        return None
    if state_dict_a is None:
        device = _get_target_device(list(state_dict_b.values()) if state_dict_b else [], target_device)
        return {k: v.clone().detach().to(device) for k, v in state_dict_b.items()} if state_dict_b else {}
    if state_dict_b is None:
        device = _get_target_device(list(state_dict_a.values()) if state_dict_a else [], target_device)
        return {k: v.clone().detach().to(device) for k, v in state_dict_a.items()} if state_dict_a else {}

    tensors_for_device_check = list(state_dict_a.values()) + list(state_dict_b.values())
    device = _get_target_device(tensors_for_device_check, target_device)

    all_keys = set(state_dict_a.keys()) | set(state_dict_b.keys())
    if not all_keys:
        return {}  # Both dicts were empty

    result_dict: StateDict = {}
    for key in all_keys:
        tensor_a = state_dict_a.get(key)
        tensor_b = state_dict_b.get(key)
        if tensor_a is not None and tensor_b is not None:
            result_dict[key] = (
                tensor_a.to(device).clone().detach() +
                tensor_b.to(device).clone().detach()
            )
        elif tensor_a is not None:
            result_dict[key] = tensor_a.to(device).clone().detach()
        elif tensor_b is not None:
            result_dict[key] = tensor_b.to(device).clone().detach()
        # If both are None, the key is skipped.
    return result_dict


def scale_state_dict(
    state_dict: Optional[StateDict],
    scalar: float,
    target_device: Optional[torch.device] = None
) -> Optional[StateDict]:
    """
    Multiplies all tensors in a state dictionary by a scalar value.

    Args:
        state_dict: The state dictionary to scale.
        scalar: The scalar value to multiply by.
        target_device: Desired device for the resulting tensors. If None,
                       inferred from the state_dict or defaults to CPU.

    Returns:
        A new state dictionary with scaled tensors, or an empty dictionary if input
        is invalid.
    """
    if state_dict is None or not state_dict:
        return {}  # Return empty dict for empty input

    device = _get_target_device(list(state_dict.values()), target_device)
    result_dict: StateDict = {}

    for key, tensor in state_dict.items():
        if tensor is not None:
            result_dict[key] = tensor.to(device).clone().detach() * scalar
        else:
            print(f"Warning: Found None tensor for key '{key}' during scaling. Skipping key.")
    return result_dict

class Client:
    """
    Federated learning client supporting multiple local-training algorithms:
    - FedAvg (local_training_avg)
    - FedProx (local_training_prox)
    - SCAFFOLD (local_training_scaffold)
    - MOON (local_training_moon)
    - Continuous distillation (local_training_continuous)
    """
    def __init__(
        self,
        args: Any,
        training_loader: Optional[DataLoader],
        grad_cal_loader: Optional[DataLoader],
        class_set: Any
    ) -> None:
        # Configuration
        self.dataset_name = getattr(args, 'dataset_name', 'unknown')
        self.algorithm = getattr(args, 'algorithm', 'fedavg')
        self.continuous = bool(getattr(args, 'continuous', False))
        self.num_steps = {
            'training': getattr(args, 'num_SGD_training', 1),
            'grad_cal': getattr(args, 'num_SGD_grad_cal', 1)
        }

        # Hyperparameters (learning rate passed per-call)
        self.hyperparams = {
            'momentum': getattr(args, 'momentum', 0.9),
            'prox_alpha': getattr(args, 'prox_alpha', 0.01),
            'moon_mu': getattr(args, 'moon_mu', 1.0),
            'moon_tau': getattr(args, 'moon_tau', 0.1),
            'kl_coefficient': getattr(args, 'kl_coefficient', 5.0),
            'acg_beta': getattr(args, 'acg_beta', 0.01),
            'acg_lambda': getattr(args, 'acg_lambda', 0.85),
        }

        # Data loaders & class set
        self.training_loader = training_loader
        self.grad_cal_loader = grad_cal_loader
        self.class_set = class_set

        # Devices
        self.gpu = torch.device(
            f"cuda:{getattr(args, 'gpu_index', 0)}" if torch.cuda.is_available() else "cpu"
        )
        self.cpu = torch.device("cpu")

        # Instantiate primary model
        self.local_model = self._build_model(args).to(self.cpu)

        self.model_last_session = self._build_model(args).to(self.cpu)
        self.model_last_session.eval()

        self._init_aux_models()

        # Placeholder for optimizer
        self.optimizer: Optional[optim.Optimizer] = None

        # Correction / regularization state
        self.scaffold_c = self._zero_state_dict(self.local_model)

    def _build_model(self, args: Any) -> nn.Module:
        
        return create_model(args)

    def _generic_resnet(self, depth: int, out_dim: int) -> nn.Module:
        model = {18: tvmodels.resnet18, 34: tvmodels.resnet34}[depth](weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, out_dim)
        return model

    def _init_aux_models(self) -> None:
        if 'moon' in self.algorithm or self.continuous:
            self.model_global = copy.deepcopy(self.local_model).to(self.cpu)
            self.model_global.eval()
        else:
            self.model_global = None

        if self.algorithm=='moon':
            self.model_prev = copy.deepcopy(self.local_model).to(self.cpu)
            self.model_prev.eval()
        else:
            self.model_prev = None

    def _zero_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: torch.zeros_like(v, dtype=torch.float32, device=self.cpu)
                for k,v in model.state_dict().items()}

    def _clone_cpu(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k,v in state.items()}

    def _load_model(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        try:
            self.local_model.load_state_dict(state_dict)
            return True
        except Exception as e:
            logging.error("Load failed: %s", e)
            traceback.print_exc()
            return False

    def _prepare_training(
        self,
        global_state: Dict[str, torch.Tensor]
    ) -> bool:
        """Load global state into model before training."""
        cpu_state = self._clone_cpu(global_state)
        if not self._load_model(cpu_state):
            return False
        self.local_model.to(self.gpu)
        return True

    def _finalize(self) -> Dict[str, torch.Tensor]:
        self.local_model.to(self.cpu)
        return {k: v.detach().cpu()
                for k,v in self.local_model.state_dict().items()}

    def _process_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        is_rnn: bool
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Any]]:
        inputs, labels = batch
        inputs, labels = inputs.to(self.gpu), labels
        hidden = None
        try:
            if is_rnn:
                bs = inputs.size(0)
                hidden = self.local_model.init_hidden(bs)  # type: ignore
                if isinstance(hidden, tuple):
                    hidden = (hidden[0].to(inputs.device), hidden[1].to(inputs.device))
                else:
                    hidden = hidden.to(inputs.device)
                logits, hidden = self.local_model(inputs, hidden)
                labels = labels.long().to(self.gpu)
            else:
                logits = self.local_model(inputs)
                labels = labels.to(self.gpu).long()
            if logits is None or logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
                return None, None, None
            return logits, labels, hidden
        except Exception as e:
            logging.error("_process_batch error: %s", e)
            traceback.print_exc()
            return None, None, None

    def _train_standard(
        self,
        criterion: nn.Module,
        num_steps: int,
        extra_loss: Optional[Callable]
    ) -> int:
        self.local_model.train()
        is_rnn = hasattr(self.local_model, 'is_rnn_like') and self.local_model.is_rnn_like  # type: ignore
        processed = 0
        for idx, batch in enumerate(self.training_loader or []):
            if idx >= num_steps or self.optimizer is None:
                break
            logits, lbls, _ = self._process_batch(batch, is_rnn)
            if logits is None:
                continue
            loss = criterion(logits, lbls)

            print("Loss:", loss.item())

            if extra_loss:
                loss = extra_loss(logits, lbls, batch[0], loss, batch[1])
            try:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                processed += 1
            except Exception as e:
                logging.error("Train step %d failed: %s", idx, e)
                traceback.print_exc()
                break
        return processed

    def _train_scaffold(
        self,
        global_state: Dict[str, torch.Tensor],
        global_c: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        proc = self._train_standard(nn.CrossEntropyLoss(), self.num_steps['training'], None)
        local_cpu = self._clone_cpu(self.local_model.state_dict())
        glob_cpu = self._clone_cpu(global_state)
        lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        K = float(max(1, proc))
        delta_y = subtract_state_dicts(local_cpu, glob_cpu, target_device=self.cpu)
        if delta_y and lr > 0:
            term = scale_state_dict(delta_y, -1.0/(K*lr), target_device=self.cpu)
            neg_g = scale_state_dict(self._clone_cpu(global_c), -1.0, target_device=self.cpu)
            delta_c = add_state_dicts(neg_g, term, target_device=self.cpu)
            self.scaffold_c = add_state_dicts(self.scaffold_c, delta_c, target_device=self.cpu)
        else:
            delta_c = None
        return delta_y, delta_c

    def _train_moon(self, global_state: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        if not self.model_global or not self.model_prev:
            return None
        cpu_state = self._clone_cpu(global_state)
        try:
            self.model_global.load_state_dict(cpu_state)
            self.model_global.eval()
        except Exception as e:
            logging.error("MOON global load failed: %s", e)
            traceback.print_exc()
            return None

        criterion = nn.CrossEntropyLoss()
        cosim = nn.CosineSimilarity(dim=-1)
        mu, tau = self.hyperparams['moon_mu'], self.hyperparams['moon_tau']
        processed = 0
        is_rnn = hasattr(self.local_model, 'is_rnn_like') and self.local_model.is_rnn_like  # type: ignore

        for idx, batch in enumerate(self.training_loader or []):
            if idx >= self.num_steps['training'] or self.optimizer is None:
                break
            logits, lbls, _ = self._process_batch(batch, is_rnn)
            if logits is None:
                continue
            l_sup = criterion(logits, lbls)
            inputs = batch[0].to(self.gpu)
            with torch.no_grad():
                self.model_global.to(self.gpu)
                zg = self.model_global.forward_features(inputs)  # type: ignore
                self.model_prev.to(self.gpu)
                zp = self.model_prev.forward_features(inputs)   # type: ignore
            zl = self.local_model.forward_features(inputs)  # type: ignore
            if zl is not None and zg is not None and zp is not None:
                pos = cosim(zl, zg)
                neg = cosim(zl, zp)
                logits_c = torch.stack([pos, neg], dim=1) / tau
                cons_lbl = torch.zeros(pos.size(0), dtype=torch.long, device=self.gpu)
                l_cons = criterion(logits_c, cons_lbl)
            else:
                l_cons = torch.tensor(0.0, device=self.gpu)
            loss = l_sup + mu * l_cons
            try:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                processed += 1
            except Exception as e:
                logging.error("MOON train step %d failed: %s", idx, e)
                traceback.print_exc()
                break

        try:
            new_cpu = self._clone_cpu(self.local_model.state_dict())
            self.model_prev.load_state_dict(new_cpu)
        except Exception:
            pass
        return self._finalize()

    def _train_continuous(self) -> Optional[Dict[str, torch.Tensor]]:
        
        if not self.model_last_session:
            return None

        self.model_last_session.eval()

        ce = nn.CrossEntropyLoss()
        kl_coeff = self.hyperparams['kl_coefficient']
        processed = 0
        is_rnn_s = hasattr(self.local_model, 'is_rnn_like') and self.local_model.is_rnn_like  # type: ignore
        is_rnn_t = hasattr(self.model_last_session, 'is_rnn_like') and self.model_last_session.is_rnn_like  # type: ignore

        for idx, batch in enumerate(self.training_loader or []):
            if idx >= self.num_steps['training'] or self.optimizer is None:
                break
            logits_s, lbls, _ = self._process_batch(batch, is_rnn_s)
            if logits_s is None:
                continue
            inputs = batch[0].to(self.gpu)
            with torch.no_grad():
                self.model_last_session.to(self.gpu)
                logits_t = self.model_last_session(inputs)  # type: ignore
            ce_loss = ce(logits_s, lbls)
            logp = F.log_softmax(logits_s, dim=1)
            pg = F.softmax(logits_t, dim=1)
            kl_loss = F.kl_div(logp, pg.detach(), reduction='batchmean')
            loss = ce_loss + kl_coeff * kl_loss
            try:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                processed += 1
            except Exception as e:
                logging.error("Continuous train step %d failed: %s", idx, e)
                traceback.print_exc()
                break

        return self._finalize()

    def set_model_last_session(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Load a new state into model_last_session (e.g. after finishing a session),
        ensuring all tensors are moved to CPU first.
        """
        # move every tensor to CPU, detach & clone
        cpu_state = self._clone_cpu(state_dict)

        try:
            self.model_last_session.load_state_dict(cpu_state)
            self.model_last_session.eval()
            return True
        except Exception as e:
            logging.error("Failed to set last-session model: %s", e)
            return False

    # Public API

    def local_training_acg(
        self,
        global_state: Dict[str, torch.Tensor],
        global_m: Dict[str, torch.Tensor],
        lr: float
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        FedACG local step:
        1. Accelerated init: θ₀ = w + λ·m
        2. K steps of proximal SGD with β regularization
        3. Return Δ = θ_K - θ₀
        """
        beta = float(self.hyperparams['acg_beta'])
        lam = float(self.hyperparams['acg_lambda'])
        # 1) build accelerated init on CPU
        accel_state_cpu = {
            k: (global_state[k].cpu() + lam * global_m[k].cpu()).detach().clone()
            for k in global_state
        }
        if not self._load_model(accel_state_cpu):
            return None
        self.local_model.to(self.gpu)
        # keep init on GPU for prox term
        accel_state_gpu = {k: v.to(self.gpu) for k, v in accel_state_cpu.items()}

        # 2) optimizer
        self.optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=lr,
            momentum=self.hyperparams['momentum']
        )

        # 3) define proximal extra loss
        def acg_prox(logits, labels, inputs, orig_loss, orig_labels):
            reg = torch.tensor(0.0, device=self.gpu)
            for name, param in self.local_model.named_parameters():
                if not param.requires_grad:
                    continue
                init_p = accel_state_gpu[name]
                reg = reg + 0.5 * beta * (param - init_p).pow(2).sum()
            return orig_loss + reg

        # 4) train K steps with prox term
        self._train_standard(nn.CrossEntropyLoss(), self.num_steps['training'], acg_prox)

        # 5) finalize and compute Δ
        local_cpu = self._clone_cpu(self.local_model.state_dict())
        delta = {k: local_cpu[k] - accel_state_cpu[k] for k in accel_state_cpu}
        return delta


    def local_training_avg(
        self,
        global_state: Dict[str, torch.Tensor],
        lr: float
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._prepare_training(global_state):
            return None
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=lr, momentum=self.hyperparams['momentum'])
        self._train_standard(nn.CrossEntropyLoss(), self.num_steps['training'], None)
        return self._finalize()

    def local_training_prox(
        self,
        global_state: Dict[str, torch.Tensor],
        lr: float
    ) -> Optional[Dict[str, torch.Tensor]]:
        def prox_term(logits, labels, inputs, loss, orig_labels):
            term = torch.tensor(0.0, device=self.gpu)
            for name, param in self.local_model.named_parameters():
                gp = global_state[name].to(self.gpu)
                term += torch.sum((param - gp) ** 2)
            return loss + 0.5 * self.hyperparams['prox_alpha'] * term

        if not self._prepare_training(global_state):
            return None
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=lr, momentum=self.hyperparams['momentum'])
        self._train_standard(nn.CrossEntropyLoss(), self.num_steps['training'], prox_term)
        return self._finalize()

    def local_training_scaffold(
        self,
        global_state: Dict[str, torch.Tensor],
        global_c: Dict[str, torch.Tensor],
        lr: float
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        if not self._prepare_training(global_state):
            return None, None
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=lr, momentum=self.hyperparams['momentum'])
        return self._train_scaffold(global_state, global_c)

    def local_training_continuous(
        self,
        global_state: Dict[str, torch.Tensor],
        lr: float
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._prepare_training(global_state):
            return None
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=lr, momentum=self.hyperparams['momentum'])
        return self._train_continuous()

    def local_training_moon(
        self,
        global_state: Dict[str, torch.Tensor],
        lr: float
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        MOON (Model-Contrastive Federated Learning):
        Prepares the model, runs MOON’s local contrastive training, and returns updated params.
        """
        if not self._prepare_training(global_state):
            return None
        self.optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=lr,
            momentum=self.hyperparams['momentum']
        )
        return self._train_moon(global_state)


def moving_average(original_list, odd_window):
    if odd_window%2 == 0: raise ValueError
    left = right = (odd_window - 1)//2
    if 2 * left >= len(original_list): raise ValueError
    new_list = [0] * (len(original_list) - 2 * left)
    for j in range(left, len(original_list) - right):
        new_list[j -left] = sum(original_list[j - left: j-left + odd_window])/odd_window
    
    copied_list = copy.deepcopy(original_list)
    copied_list[left: left + len(new_list)] = new_list

    return copied_list

def moving_average_reduce(original_list, window):
    if window > len(original_list): raise ValueError("The window is larger than the size of the original list")
    new_list = [0] * (len(original_list) - (window -1))
    for i in range(len(original_list) - (window -1)):
        new_list[i] = sum(original_list[i: i+window])/window
    return new_list


@torch.no_grad() # Keep no_grad for efficiency during inference
def test_inference(args, device, model, testloader):
    """
    Returns the test accuracy and average loss.
    Handles different model/data types correctly.

    Args:
        args: An object that should contain the attribute 'dataset_name'.
        device: The torch device to run inference on.
        model: The PyTorch model (can be CharRNN or other types).
        testloader: DataLoader for the test set.

    Returns:
        accuracy: The fraction of correctly predicted samples.
        avg_loss: The average loss per sample.
    """
    model.eval() # Set model to evaluation mode
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss() # Use standard CrossEntropyLoss
    total_loss = 0.0
    correct = 0
    total_elements = 0 # Tracks total items evaluated (samples in this context)

    # Check if the model is CharRNN (or needs specific RNN handling)
    # Make this check more robust if you have multiple RNN types
    is_rnn = hasattr(model, 'init_hidden') and callable(model.init_hidden)
    hidden = None # Initialize hidden state placeholder for RNN

    if testloader is None or testloader.dataset is None or len(testloader.dataset) == 0:
        print("Warning: test_inference received empty or invalid loader.")
        return 0.0, 0.0

    for batch_idx, batch_data in enumerate(testloader):
        try:
            # Assuming your loader *always* yields (data, labels) now
            # due to the consistent use of collate_fn when needed
            data, labels = batch_data
        except Exception as e:
            print(f"Error unpacking batch {batch_idx}: {e}. Skipping.")
            continue

        try:
            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)
            if batch_size == 0: continue
        except Exception as e:
            print(f"Error moving data to device for batch {batch_idx}: {e}. Skipping.")
            continue

        # --- CORRECTED Conditional Data Type Handling ---
        # Default: Assume data is LongTensor (for embeddings) unless specified otherwise
        # Default: Assume labels should be LongTensor for CrossEntropyLoss

        # Specific overrides based on dataset/model type:
        if not is_rnn:
            # Define datasets that *require* float input data (e.g., image, tabular)
            # Add names of datasets that truly need float input for their models
            datasets_requiring_float_input = ["mnist", 'fmnist', 'SVHN' ,'cifar10', 'cifar100', 'tinyimagenet'] # e.g., ["cifar10", "some_tabular_data"]

            if args.dataset_name in datasets_requiring_float_input:
                try:
                    # Only cast to float if absolutely necessary
                    if data.dtype != torch.float:
                       data = data.float()
                except RuntimeError as e:
                    print(f"Warning: Failed data.float() for known float dataset ({args.dataset_name}). Error: {e}")
            # else: data remains LongTensor (correct for AG News + TextCNN)

        # Ensure labels are Long for CrossEntropyLoss, regardless of model type
        if labels.dtype != torch.long:
            try:
                # Special handling for specific datasets if needed *before* casting
                # if args.dataset_name == "some_dataset_with_odd_labels":
                #     labels = preprocess_special_labels(labels)
                labels = labels.long()
            except Exception as label_err:
                 print(f"Warning: Failed to convert labels to long for batch {batch_idx}. Error: {label_err}. Skipping.")
                 continue
        # --- End CORRECTED Conditional Data Type Handling ---


        try:
            # --- Conditional Forward Pass ---
            if is_rnn:
                # Assumes model is the modified CharRNN outputting [BatchSize, VocabSize]
                hidden = model.init_hidden(batch_size) # Assuming device handled inside init_hidden or model transfers it
                outputs, hidden = model(data, hidden)
            else:
                # Standard forward pass (e.g., TextCNN)
                # Input `data` should be LongTensor here for AG News
                outputs = model(data)
            # --- End Conditional Forward Pass ---

            # --- Loss and Accuracy Calculation ---
            # Expecting outputs: [N, C], labels: [N] where N=BatchSize
            if outputs.ndim != 2 or labels.ndim != 1 or outputs.shape[0] != labels.shape[0]:
                 print(f"Warning: Mismatch outputs ({outputs.shape}) vs labels ({labels.shape}) in eval batch {batch_idx}. Skipping.")
                 continue

            batch_loss = criterion(outputs, labels)
            current_batch_elements = labels.size(0) # Use label count
            total_loss += batch_loss.item() * current_batch_elements

            _, pred_labels = torch.max(outputs, dim=1)
            correct += (pred_labels == labels).sum().item()
            total_elements += current_batch_elements
            # --- End Metric Calculation ---

        except Exception as forward_err:
            print(f"Error during forward pass or metric calculation for batch {batch_idx}: {forward_err}")
            traceback.print_exc()
            continue # Skip this batch on error

    # --- Final Calculation ---
    accuracy = correct / total_elements if total_elements > 0 else 0
    avg_loss = total_loss / total_elements if total_elements > 0 else 0

    return accuracy, avg_loss

def compute_client_weights(client_list, current_device_idx_list, loader_name="training_loader"):
    """
    Compute FedAvg weights based on the size of a chosen dataloader.

    Args:
        client_list: list of Client objects
        current_device_idx_list: list of indices into client_list
        loader_name: str, either "training_loader" or "grad_cal_loader"

    Returns:
        dict mapping client index → fraction of total samples
    """
    if loader_name not in ("training_loader", "grad_cal_loader"):
        raise ValueError(f"loader_name must be 'training_loader' or 'grad_cal_loader', got {loader_name!r}")

    # 1) sum up all samples across the chosen loader
    total_num_data = 0
    for i in current_device_idx_list:
        loader = getattr(client_list[i], loader_name)
        total_num_data += len(loader.dataset)

    # 2) compute each client's share
    return {
        i: len(getattr(client_list[i], loader_name).dataset) / total_num_data
        for i in current_device_idx_list
    }


def filter_dataset_by_classes(dataset, class_set, batch_size, dataset_name=None):
    """
    Filter a PyTorch-style or Hugging Face dataset by class labels.

    Args:
        dataset: A PyTorch Dataset (with .labels or .targets) or a Hugging Face Dataset
                 (indexable with dataset["label"]).
        class_set: A set of integer class labels to keep.
        batch_size: Batch size for the DataLoader.
        dataset_name: Optional string to select a custom collate_fn (e.g. "ag_news").

    Returns:
        A DataLoader over only those samples whose label is in class_set.
    """
    # 1) Extract labels array
    if hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "targets"):
        labels = dataset.targets
    else:
        try:
            labels = dataset["label"]
        except (KeyError, TypeError):
            raise ValueError("Could not extract 'label' field from dataset.")

    # 2) Convert labels to a plain Python list
    if isinstance(labels, torch.Tensor) or isinstance(labels, np.ndarray):
        labels = labels.tolist()

    # 3) Find all indices whose label is in class_set
    filtered_indices = [i for i, lbl in enumerate(labels) if lbl in class_set]
    if not filtered_indices:
        raise ValueError(f"No samples found for class labels: {class_set}")

    # 4) Optionally pick up a custom collate function
    collate_fn_mapping = {
        "ag_news": dict_to_tuple_collate
    }
    collate_fn = collate_fn_mapping.get(dataset_name, None)

    # 5) Build and return the DataLoader
    subset = Subset(dataset, filtered_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def _extract_targets(dataset):
    """
    Extracts integer class labels from a variety of dataset types in an efficient, unified way.

    Supported sources (in precedence order):
      1. `dataset.targets` attribute (e.g., torchvision datasets).
      2. `dataset.labels` attribute (legacy/custom datasets).
      3. Hugging Face Datasets: `dataset.features['label']` and `dataset['label']`.
      4. Fallback via `__getitem__` iteration:
         - If `dataset[0]` is a dict with a 'label' key.
         - If `dataset[0]` is a tuple/list where index 1 is the label.

    Returns:
        np.ndarray of shape (N,) and dtype int, where N = len(dataset).
        Each entry is the integer class label of the corresponding example.

    Raises:
        ValueError:
            - If no `.targets`, `.labels`, or `features['label']` is found.
            - If fallback via `__getitem__` cannot infer the label position.
    """
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=int)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=int)

    features = getattr(dataset, "features", {})
    if isinstance(features, dict) and "label" in features:
        label_data = dataset["label"]
        if isinstance(label_data, (list, np.ndarray)):
            return np.asarray(label_data, dtype=int)
        if hasattr(label_data, "numpy"):
            arr = label_data.cpu().numpy() if hasattr(label_data, "cpu") else label_data.numpy()
            return arr.astype(int)
        return np.asarray(list(label_data), dtype=int)

    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        first = dataset[0]
        N = len(dataset)
        if isinstance(first, dict) and "label" in first:
            return np.asarray([dataset[i]["label"] for i in range(N)], dtype=int)
        if isinstance(first, (tuple, list)) and len(first) > 1:
            return np.asarray([dataset[i][1] for i in range(N)], dtype=int)
        raise ValueError("Cannot infer label from dataset[0].")

    raise ValueError(
        "Dataset has no `.targets`/`.labels`, no `features['label']`, "
        "and `__getitem__` fallback failed."
    )


def partition_and_schedule(
    dataset,
    num_clients_per_group: int,
    num_sessions: int,
    num_sessions_pilot: int,
    num_rounds_pilot: int,
    num_rounds_actual: int,
    cross_session_label_overlap: float,
    in_session_label_dist: str,
    dirichlet_alpha: float = 0.5,
    unbalanced_sgm: float = 0.0
):
    """
    Partitions a dataset for federated learning into two client-groups (A, B),
    splits the label space into two subsets (with optional overlap between groups for different sessions),
    allocates samples to clients within a session/group based on a Dirichlet distribution, and builds:
      • `session_clients`: list of length `num_sessions`, alternating A/B.
      • `training_order`: fully expanded per-round schedule.

    Clients:
      - Total = 2 * num_clients_per_group
      - A = [0 .. num_clients_per_group-1]
      - B = [num_clients_per_group .. 2*num_clients_per_group-1]

    Label splitting (determines labels available to Group A vs. Group B across sessions):
      - If `cross_session_label_overlap == 0`: non-overlapping half-split of L labels.
      - If 0 < `cross_session_label_overlap` < 1:
          G = ⌊L / (2 - cross_session_label_overlap)⌋
          O = ⌊G * cross_session_label_overlap⌋
          If O == 0, fallback to half-split.
          Else:
            U = G - O
            uniq1 = labels[0:U]
            overl = labels[U:U+O]
            uniq2 = labels[U+O:U+O+U]
            labels1 (for Group A) = uniq1 + overl
            labels2 (for Group B) = overl + uniq2
      Unused labels remain unassigned.

    In-session label distribution (`in_session_label_dist`):
      - "dirichlet":
          1. Determine per-client quotas: distribute total samples among clients
             (balanced or log-normal if `unbalanced_sgm > 0`).
          2. For each client, draw class proportions from Dirichlet(`dirichlet_alpha`).
          3. Assign samples by inverse‐CDF sampling on those priors, falling back
             to any remaining class when necessary.

    Args:
        dataset:
            Any dataset with integer labels accessible via `_extract_targets`.
        num_clients_per_group (int):
            Clients per group; total federated clients = 2 * this value.
        num_sessions (int):
            Number of sessions to alternate between A and B.
        num_sessions_pilot (int):
            First N sessions use `num_rounds_pilot`.
        num_rounds_pilot (int):
            Rounds per session in the pilot phase.
        num_rounds_actual (int):
            Rounds per session in the actual phase.
        cross_session_label_overlap (float):
            Fraction of label overlap between Group A's label set and Group B's.
        in_session_label_dist (str):
            Method for distributing samples; only "dirichlet" is supported.
        dirichlet_alpha (float):
            α parameter for the Dirichlet distribution.
        unbalanced_sgm (float):
            Sigma for log-normal quotas when using Dirichlet. If 0, quotas are balanced.

    Returns:
        clients_data_ids (dict[int, list[int]]):
            Mapping from client_id to assigned sample indices.
        client_classes (dict[int, set[int]]):
            Mapping from client_id to its set of labels.
        session_clients (list[list[int]]):
            Length = num_sessions; each entry is the client group for that session.
        training_order (list[list[int]]):
            Per-round client IDs; total rounds =
            num_sessions_pilot*num_rounds_pilot +
            (num_sessions-num_sessions_pilot)*num_rounds_actual

    Raises:
        ValueError:
          - If `num_clients_per_group < 1`.
          - If `num_sessions_pilot > num_sessions`.
          - If `cross_session_label_overlap` not in [0,1).
          - If `in_session_label_dist` is not "dirichlet".
          - If `unbalanced_sgm` is negative.
    """
    # --- Argument Validation ---
    if num_clients_per_group < 1:
        raise ValueError("num_clients_per_group must be ≥ 1.")
    if num_sessions_pilot > num_sessions:
        raise ValueError("num_sessions_pilot cannot exceed num_sessions.")
    if not (0 <= cross_session_label_overlap < 1):
        raise ValueError("cross_session_label_overlap must be in [0,1).")
    # Only Dirichlet allocation is supported now
    if in_session_label_dist not in ("dirichlet",):
        raise ValueError("in_session_label_dist must be 'dirichlet'.")
    if unbalanced_sgm < 0:
        raise ValueError("unbalanced_sgm must be non-negative.")

    # --- 1. Extract Labels from Dataset ---
    targets = _extract_targets(dataset)
    unique_labels = np.unique(targets)
    L = unique_labels.size

    # --- 2. Define Client Groups ---
    total_clients = 2 * num_clients_per_group
    clients = list(range(total_clients))
    groupA = clients[:num_clients_per_group]
    groupB = clients[num_clients_per_group:]

    # --- 3. Split Labels for Group A and Group B ---
    if L == 0:
        labels1, labels2 = [], []
    elif cross_session_label_overlap == 0:
        half = L // 2
        labels1 = unique_labels[:half].tolist()
        labels2 = unique_labels[half:].tolist()
    else:
        denom = (2 - cross_session_label_overlap)
        G = int(np.floor(L / denom)) if denom != 0 else L
        O = int(np.floor(G * cross_session_label_overlap))
        if O == 0 or G == 0 or G <= O:
            half = L // 2
            labels1 = unique_labels[:half].tolist()
            labels2 = unique_labels[half:].tolist()
        else:
            U = G - O
            uniq1 = unique_labels[:U]
            overl = unique_labels[U:U+O]
            uniq2 = unique_labels[U+O:U+O+U]
            labels1 = sorted(set(uniq1.tolist() + overl.tolist()))
            labels2 = sorted(set(overl.tolist() + uniq2.tolist()))

    print(f"[DEBUG] Labels for Group A (labels1): {labels1} (Count: {len(labels1)})")
    print(f"[DEBUG] Labels for Group B (labels2): {labels2} (Count: {len(labels2)})")

    # --- 4. Allocate Data Samples to Clients via Dirichlet ---
    clients_data_ids = {cid: [] for cid in clients}
    client_classes   = {cid: set() for cid in clients}

    if in_session_label_dist == "dirichlet":
        for lbls_for_group, grp_client_ids in ((labels1, groupA), (labels2, groupB)):
            num_clients_in_grp = len(grp_client_ids)
            if num_clients_in_grp == 0 or not lbls_for_group:
                continue

            group_label_indices = {}
            total_samples_for_group_labels = 0
            for lbl in lbls_for_group:
                idxs = np.where(targets == lbl)[0]
                group_label_indices[lbl] = idxs
                total_samples_for_group_labels += len(idxs)

            if total_samples_for_group_labels == 0:
                continue

            # Determine per-client quotas
            avg = total_samples_for_group_labels / num_clients_in_grp
            if unbalanced_sgm > 0:
                mu = np.log(avg) if avg > 1e-9 else -10
                raw = np.random.lognormal(mean=mu, sigma=unbalanced_sgm, size=num_clients_in_grp)
                quotas = ((raw / raw.sum()) * total_samples_for_group_labels).astype(int)
            else:
                quotas = np.full(num_clients_in_grp, avg, dtype=int)

            diff = total_samples_for_group_labels - quotas.sum()
            for i in range(abs(diff)):
                quotas[i % num_clients_in_grp] += np.sign(diff)
            quotas = np.clip(quotas, 0, None)

            # Draw Dirichlet priors
            client_prior_cumsum_map = {
                cid: np.cumsum(np.random.dirichlet([dirichlet_alpha] * len(lbls_for_group)))
                for cid in grp_client_ids
            }

            # Prepare pools
            indices_by_label_pool = {
                lbl: list(group_label_indices[lbl]) for lbl in lbls_for_group
            }
            for lbl in indices_by_label_pool:
                np.random.shuffle(indices_by_label_pool[lbl])
            samples_left_in_label_pool = {
                lbl: len(indices_by_label_pool[lbl]) for lbl in lbls_for_group
            }

            # Assign samples to each client
            for client_idx, current_client_id in enumerate(grp_client_ids):
                need = quotas[client_idx]
                if need <= 0:
                    continue
                cdf = client_prior_cumsum_map[current_client_id]

                for _ in range(need):
                    if not any(samples_left_in_label_pool.values()):
                        break
                    # sample by prior
                    for _ in range(len(lbls_for_group) * 3 + 5):
                        u = np.random.rand()
                        chosen_label = lbls_for_group[np.argmax(u <= cdf)]
                        if samples_left_in_label_pool[chosen_label] > 0:
                            sample_idx = indices_by_label_pool[chosen_label].pop()
                            clients_data_ids[current_client_id].append(sample_idx)
                            client_classes[current_client_id].add(chosen_label)
                            samples_left_in_label_pool[chosen_label] -= 1
                            break
                    else:
                        # fallback to any remaining label
                        available = [l for l in lbls_for_group if samples_left_in_label_pool[l] > 0]
                        if not available:
                            break
                        lbl = np.random.choice(available)
                        sample_idx = indices_by_label_pool[lbl].pop()
                        clients_data_ids[current_client_id].append(sample_idx)
                        client_classes[current_client_id].add(lbl)
                        samples_left_in_label_pool[lbl] -= 1

    # --- 5. Define Sessions and Training Order ---
    session_clients = [
        groupA.copy() if (s % 2 == 0) else groupB.copy()
        for s in range(num_sessions)
    ]
    training_order = []
    for s, sess_grp in enumerate(session_clients):
        reps = num_rounds_pilot if s < num_sessions_pilot else num_rounds_actual
        training_order += [sess_grp.copy()] * reps

    return clients_data_ids, client_classes, session_clients, training_order


def zero_state_dict_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state-dict–shaped dict of float32 zeros on the same device as model.
    """
    return {
        name: torch.zeros_like(param, dtype=torch.float32, device="cpu")
        for name, param in model.state_dict().items()
    }

def get_lr_schedule_from_config(args: Namespace) -> Tuple[float, float]:
    """
    Load and return (initial_lr, eta_min) for a given dataset and algorithm
    from a JSON file.

    Expects:
      - args.lr_config_path (str or Path)
      - args.dataset_name (str)
      - args.algorithm (str)

    Raises:
        FileNotFoundError: config file is missing.
        ValueError: invalid or missing arguments, or keys not found.
        json.JSONDecodeError: invalid JSON format.
        TypeError: non-numeric or wrongly shaped entry in config.
    """
    # Validate config path
    try:
        config_path = Path(args.lr_config_path)
    except Exception:
        raise ValueError("Must provide `--lr_config_path` pointing to a JSON file.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path!s}")

    # Validate required args
    ds_name = getattr(args, "dataset_name", None)
    alg_name = getattr(args, "algorithm", None)
    if not ds_name or not alg_name:
        raise ValueError("Both `dataset_name` and `algorithm` must be specified in args.")

    # Load JSON
    with config_path.open("r") as f:
        lr_config = json.load(f)

    # Lookup dataset
    ds_cfg = lr_config.get(ds_name)
    if ds_cfg is None:
        available = ", ".join(lr_config.keys())
        raise ValueError(f"Dataset '{ds_name}' not in config (available: {available}).")

    # Lookup algorithm
    lr_value = ds_cfg.get(alg_name)
    if lr_value is None:
        available = ", ".join(ds_cfg.keys())
        raise ValueError(
            f"Algorithm '{alg_name}' not defined for dataset '{ds_name}' "
            f"(available: {available})."
        )

    # Validate shape
    if not isinstance(lr_value, (list, tuple)) or len(lr_value) != 2:
        raise TypeError(
            f"Expected a list or tuple of two numbers for "
            f"{ds_name}/{alg_name}; got {lr_value!r}."
        )

    # Validate contents
    lr, eta_min = lr_value
    for name, val in (("lr", lr), ("eta_min", eta_min)):
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"Expected numeric {name} for {ds_name}/{alg_name}; "
                f"got {type(val).__name__}."
            )

    return float(lr), float(eta_min)


def require_init_lr(args) -> float:
    """
    Fetch args.init_lr or fail with a clear message.
    """
    try:
        init_lr = args.init_lr
    except AttributeError:
        raise AttributeError(
            "Missing `args.init_lr`—make sure you called get_learning_rate_from_config(args) before."
        )
    if not isinstance(init_lr, float):
        raise TypeError(f"Expected args.init_lr to be a float, got {type(init_lr).__name__}")
    return init_lr

def require_lr_min_ratio(args) -> float:
    """
    Fetch args.lr_min_ratio or fail with a clear message.
    """
    try:
        lr_min_ratio = args.lr_min_ratio
    except AttributeError:
        raise AttributeError(
            "Missing `args.lr_min_ratio`—make sure you called "
            "`get_lr_schedule_from_config(args)` before."
        )
    if not isinstance(lr_min_ratio, float):
        raise TypeError(
            f"Expected args.lr_min_ratio to be a float, got {type(lr_min_ratio).__name__}"
        )
    return lr_min_ratio

def create_model(args: Any) -> nn.Module:
    """
    Creates and returns a model instance based on the dataset name and algorithm
    derived entirely from the args object.

    Args:
        args: An object containing various arguments, including 'dataset_name',
              'algorithm', 'vocab_size', and 'pad_idx'.

    Returns:
        An instance of a PyTorch nn.Module.

    Raises:
        AttributeError: If required attributes like 'dataset_name' or 'algorithm'
                        are missing from the args object.
        NotImplementedError: If no model is defined for the given dataset name
                             and the fallback 'net' function fails.
    """
    # Get dataset_name and algorithm directly from args
    # Using getattr with a default value can make it safer if args might be missing attributes
    dataset_name = getattr(args, 'dataset_name')
    algorithm = getattr(args, 'algorithm', 'fedavg') # Provide a default for algorithm

    if dataset_name == "ag_news":
        # Access vocab_size and pad_idx directly from args
        vocab_size = getattr(args, 'vocab_size')
        pad_idx = getattr(args, 'pad_idx')
        # return TextCNN(
        #     vocab_size=vocab_size,
        #     embed_dim=128,
        #     num_classes=4,
        #     pad_idx=pad_idx,
        # )

        return FastText(vocab_size = vocab_size, embed_dim = 300, num_classes =4, pad_idx = pad_idx)

    elif dataset_name == "SVHN":
        # Use the algorithm obtained from args
        # if algorithm == "moon":
        #     return SVHN_Cons(hidden_dims=[200, 100], projection_dim=256, output_dim=10)
        # else:
        #     return CNN_SVHN()

        # return SVHNFastContrastiveNet(
        #     projection_dim = 128,
        #     num_classes = 10,
        #     dropout_prob = 0.5
        # )
        return SVHNEfficient(output_dim = 10)
        # return MobileNetV3SmallFeatureExtractor(
        #     num_classes = 10,
        #     projection_dim =  256,
        #     pretrained  = True,
        # )


    elif dataset_name == "cifar10":

        return ResNetFeatureExtractor(resnet_version=18, num_classes=10, projection_dim=256, pretrained=True)

    elif dataset_name == "cifar100":

        return ResNetFeatureExtractor(resnet_version=18, num_classes=100, projection_dim=256, pretrained=True)
    
    elif dataset_name == "tinyimagenet":
        
        return ResNetFeatureExtractor(resnet_version=34, num_classes=200, projection_dim=256, pretrained=True)
    else:
        # Fallback to the generic net function, handling potential errors
        try:
            return net(dataset_name)
        except Exception as e:
            raise NotImplementedError(f"No model defined or creation failed for dataset '{dataset_name}': {e}")
    

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


def initial_model_construction(args, client_list, sync_idx, num_changes_in_set_of_device, change_point_indicator, init_warm_global_model_weight, average_warm_model_as_pilot_model_dict, current_client_idx_list, client_weight_grad_cal, computed_grads, global_model, similarity_normalized_value_check_dict, similarity_original_value_check_dict):

    num_sessions_pilot = args.num_sessions_pilot
    
    if num_changes_in_set_of_device == num_sessions_pilot and change_point_indicator:

        # Take average of the warm model to be the pilot model
       for k, weight_tensor in init_warm_global_model_weight.items():
        if not weight_tensor.dtype.is_floating_point:
            continue
        else:
            average_warm_model_as_pilot_model_dict[k] = weight_tensor[:num_sessions_pilot].mean(dim=0)
        
    # Compute Gradient
    if change_point_indicator and num_changes_in_set_of_device >= num_sessions_pilot:
        
        idx_after_pilot = num_changes_in_set_of_device - num_sessions_pilot

        footprint_dict = copy.deepcopy(average_warm_model_as_pilot_model_dict)

        current_lr = require_init_lr(args)/2

        alg_global_var_dict = {args.algorithm: zero_state_dict_cpu(global_model)}

        for _ in range(args.num_round_grad_cal):

            accum = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in footprint_dict.items()}

            if args.algorithm == "scaffold":
                accum_y = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in footprint_dict.items() }
                accum_c = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in alg_global_var_dict[args.algorithm].items() }

            #  Loop through clients and accumulate
            if args.algorithm == "avg":

                for client_idx in current_client_idx_list:
                    print(f'Client Index {client_idx}')

                    local_dict = client_list[client_idx].local_training_avg(
                        footprint_dict, current_lr
                    )
                    w = client_weight_grad_cal[client_idx]

                    # in-place: accum[k] += w * local_dict[k]
                    for k, v_local in local_dict.items():
                        accum[k].add_(v_local, alpha=w)
                
                footprint_dict = accum
            
            elif args.algorithm == "prox":
                
                for client_idx in current_client_idx_list:
                    print(f'Client Index {client_idx}')

                    local_dict = client_list[client_idx].local_training_prox(
                        footprint_dict, current_lr
                    )
                    w = client_weight_grad_cal[client_idx]

                    # in-place: accum[k] += w * local_dict[k]
                    for k, v_local in local_dict.items():
                        accum[k].add_(v_local, alpha=w)
                
                footprint_dict = accum
            
            elif args.algorithm == "scaffold":

                for client_idx in current_client_idx_list:
                    print(f'Client Index {client_idx}')

                    local_dy, local_dc = client_list[client_idx].local_training_scaffold(
                        footprint_dict,
                        alg_global_var_dict[args.algorithm], current_lr
                    )
                    w = client_weight_grad_cal[client_idx]

                    # in-place fused mul+add
                    for k, dy in local_dy.items():
                        accum_y[k].add_(dy, alpha=w)
                    for k, dc in local_dc.items():
                        accum_c[k].add_(dc, alpha=w)

                for k in alg_global_var_dict[args.algorithm]:
                    alg_global_var_dict[args.algorithm][k].add_(accum_c[k])
                    footprint_dict[k].add_(accum_y[k], alpha=current_lr)
            
            elif args.algorithm == "moon":

                for client_idx in current_client_idx_list:
                    print(f'Client Index {client_idx}')

                    local_dict = client_list[client_idx].local_training_moon(
                        footprint_dict, current_lr
                    )
                    w = client_weight_grad_cal[client_idx]

                    # in-place: accum[k] += w * local_dict[k]
                    for k, v_local in local_dict.items():
                        accum[k].add_(v_local, alpha=w)
                
                footprint_dict = accum
            
            elif args.algorithm == "acg":

                delta_bar = {k: torch.zeros_like(v, dtype=torch.float32) for k,v in footprint_dict.items()}

                for client_idx in current_client_idx_list:
                    delta_c = client_list[client_idx].local_training_acg(
                        footprint_dict, alg_global_var_dict[args.algorithm], current_lr
                    )               
                    w = client_weight_grad_cal[client_idx]

                    for k, p_c in delta_c.items():
                        delta_bar[k].add_(p_c, alpha=w)
    
                for k in alg_global_var_dict[args.algorithm]:
                    alg_global_var_dict[args.algorithm][k].mul_(getattr(args, 'acg_lambda', 0.85))
                    alg_global_var_dict[args.algorithm][k].add_(delta_bar[k])

                # 3) form new global state on CPU
                for k in footprint_dict:
                    footprint_dict[k].add_(alg_global_var_dict[args.algorithm][k])
            
        inv_lr = 1.0 
        num_computed_grads = args.num_sessions - num_sessions_pilot
        # guard against out-of-bounds just once
        if 0 <= idx_after_pilot < num_computed_grads:
            # alias for speed
            fdict = footprint_dict
            adict = average_warm_model_as_pilot_model_dict

            # for each parameter, compute (f – a)/lr and memcpy it into the buffer
            for name, history in computed_grads.items():
                history[idx_after_pilot].copy_(
                    (fdict[name] - adict[name]) * inv_lr
                )
        else:
            raise IndexError(f"idx_after_pilot {idx_after_pilot} out of range [0, {num_computed_grads})")

    # Initial Model Construction
    if num_changes_in_set_of_device >= (1 + num_sessions_pilot) and change_point_indicator:

        # 1) Determine your reference index
        idx_after_pilot = num_changes_in_set_of_device - num_sessions_pilot

        # 2) Pre-pick the similarity fns
        if args.similarity == "inner_product":
            sim_fn       = lambda a, b: (a * b).sum()
            normalize_fn = lambda x: x / x.norm()
            reducer      = torch.nn.Softmax(dim=0)
        elif args.similarity == "two_norm":
            sim_fn       = lambda a, b: (a - b).norm(p=2)
            normalize_fn = lambda x: x / x.norm()
            reducer      = torch.nn.Softmin(dim=0)
        else:
            raise ValueError("similarity must be 'inner_product' or 'two_norm'")

        # 3) Which params to compare? (drop biases)
        param_names = [n for n in computed_grads.keys() if "bias" not in n]

        # 4) Slice & normalize the reference (“idx_after_pilot”) once
        normalized_ref = {
            name: normalize_fn(computed_grads[name][idx_after_pilot])
            for name in param_names
        }

        # 5) Compute one mean‐similarity per earlier pilot
        similarities = [
            torch.stack([
                sim_fn(
                    normalize_fn(computed_grads[name][p]),
                    normalized_ref[name]
                )
                for name in param_names
            ]).mean()
            for p in range(idx_after_pilot)
        ]

        # 6) Pack into a tensor and normalize across pilots
        similarity_tensor = torch.stack(similarities)
        similarity_tensor = reducer(similarity_tensor * args.similarity_scale)

        # 7) Record for debugging/inspection
        similarity_normalized_value_check_dict[sync_idx] = similarity_tensor.cpu().tolist()
        similarity_original_value_check_dict[sync_idx]   = torch.stack(similarities).cpu().tolist()

        # 8) Blend historical “warm” global models on CPU
        store_global_model_dict = {
            k: torch.zeros_like(v.detach().cpu(), dtype=torch.float32) 
            for k, v in global_model.state_dict().items()
        }

        offset = num_sessions_pilot
        for i, weight in enumerate(similarity_tensor):
            idx = offset + i
            for k, batched in init_warm_global_model_weight.items():
                # batched[idx] is the “warm_global_model” for session idx
                store_global_model_dict[k].add_(batched[idx], alpha=weight)

        # 8) Load everything back into your GPU model (handles CPU→GPU under the hood)
        global_model.load_state_dict(store_global_model_dict)

def run_pilot_stage(args, client_list, pilot_stage, global_model, init_warm_global_model_weight, accuracy_list, type_testing_dataset_dict):

    num_rounds_pilot = args.num_rounds_pilot
    num_sessions_pilot = args.num_sessions_pilot
    num_rounds_actual = args.num_rounds_actual

    for sync_idx, current_client_idx_list in enumerate(pilot_stage):

        client_weight_training = compute_client_weights(client_list, current_client_idx_list, "training_loader")

        if sync_idx < num_rounds_pilot * num_sessions_pilot:
            num_changes_in_set_of_device = sync_idx//num_rounds_pilot
        else:
            num_changes_in_set_of_device = (sync_idx - num_rounds_pilot * num_sessions_pilot)//num_rounds_actual + num_sessions_pilot

        change_point_indicator = False

        if sync_idx < num_rounds_pilot * num_sessions_pilot:
            if sync_idx%num_rounds_pilot == 0:
                change_point_indicator = True
        else:
            if (sync_idx - num_rounds_pilot * num_sessions_pilot)%num_rounds_actual == 0:
                change_point_indicator = True

        if change_point_indicator:

            lr_list = calculate_polynomial_decay_lr_schedule(
                initial_client_lr = require_init_lr(args),
                total_communication_rounds = args.num_rounds_pilot,
                power = 0.9,
                min_client_lr = require_init_lr(args) * require_lr_min_ratio(args)
            )

            alg_global_var_dict = {args.algorithm: zero_state_dict_cpu(global_model)}


        current_lr = lr_list[sync_idx%args.num_rounds_pilot]

        print("")
        print("sync_idx", sync_idx, "current_lr", current_lr)

        unique_classes = set()

        # 1) pull your global parameters down to CPU once
        cpu_state_dict = {
            k: v.detach().cpu()
            for k, v in global_model.state_dict().items()
        }

        # 2) init an accumulator of the same shape on CPU

        new_global = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in cpu_state_dict.items()
        }

        if args.algorithm == "avg":

            for client_idx in current_client_idx_list:
                
                print(f'Client Index {client_idx}')
                local_model_parameter_dict = client_list[client_idx].local_training_avg(global_model.state_dict(), current_lr)
                
                w = client_weight_training[client_idx]

                for k, v_local in local_model_parameter_dict.items():
                    # new_global[k] += w * v_local  (all on CPU)
                    new_global[k].add_(v_local, alpha=w)

                unique_classes.update(client_list[client_idx].class_set)
        
        elif args.algorithm == "prox":

            for client_idx in current_client_idx_list:
            
                print(f'Client Index {client_idx}')
                local_model_parameter_dict = client_list[client_idx].local_training_prox(global_model.state_dict(), current_lr)
                
                w = client_weight_training[client_idx]

                for k, v_local in local_model_parameter_dict.items():
                    # new_global[k] += w * v_local  (all on CPU)
                    new_global[k].add_(v_local, alpha=w)

                unique_classes.update(client_list[client_idx].class_set)
        
        elif args.algorithm == "scaffold":
             
            accum_y = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in cpu_state_dict.items() }
            accum_c = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in cpu_state_dict.items() }

            # loop once over all clients, doing both accumulations
            for client_idx in current_client_idx_list:
                print(f'Client Index {client_idx}')
                local_dy, local_dc = client_list[client_idx].local_training_scaffold(
                    global_model.state_dict(), alg_global_var_dict[args.algorithm], current_lr
                )
                w = client_weight_training[client_idx]

                # in-place add: accum += w * local_delta
                for k, delta_y in local_dy.items():
                    accum_y[k].add_(delta_y, alpha=w)
                for k, delta_c in local_dc.items():
                    accum_c[k].add_(delta_c, alpha=w)

                unique_classes.update(client_list[client_idx].class_set)
            
            for k, delta in accum_c.items():
                alg_global_var_dict[args.algorithm][k].add_(delta)
            
            new_global = {
                k: cpu_state_dict[k] + current_lr * accum_y[k]
                for k in cpu_state_dict
            }
            
        elif args.algorithm == "moon":

            for client_idx in current_client_idx_list:
                
                print(f'Client Index {client_idx}')
                local_model_parameter_dict = client_list[client_idx].local_training_moon(global_model.state_dict(), current_lr)
                
                w = client_weight_training[client_idx]

                for k, v_local in local_model_parameter_dict.items():
                    # new_global[k] += w * v_local  (all on CPU)
                    new_global[k].add_(v_local, alpha=w)

                unique_classes.update(client_list[client_idx].class_set)
        
        elif args.algorithm == "acg":

            delta_bar = {k: torch.zeros_like(v, dtype=torch.float32) for k,v in cpu_state_dict.items()}

            for client_idx in current_client_idx_list:
                delta_c = client_list[client_idx].local_training_acg(
                    global_model.state_dict(), alg_global_var_dict[args.algorithm], current_lr
                )               
                w = client_weight_training[client_idx]

                unique_classes.update(client_list[client_idx].class_set)

                for k, p_c in delta_c.items():
                    delta_bar[k].add_(p_c, alpha=w)

            for k in alg_global_var_dict[args.algorithm]:
                alg_global_var_dict[args.algorithm][k].mul_(getattr(args, 'acg_lambda', 0.85))
                alg_global_var_dict[args.algorithm][k].add_(delta_bar[k])

            # 3) form new global state on CPU
            new_global = {
                k: v.cpu().clone() + alg_global_var_dict[args.algorithm][k] for k, v in global_model.state_dict().items()
            }
            
        save_warm_model_indicator = False

        if sync_idx < num_rounds_pilot * num_sessions_pilot:
            if sync_idx%num_rounds_pilot == num_rounds_pilot - 1:
                save_warm_model_indicator = True
        else:
            if (sync_idx - num_rounds_pilot * num_sessions_pilot)%num_rounds_actual == num_rounds_actual - 1:
                save_warm_model_indicator = True
        
        if save_warm_model_indicator:

            if 0 <= num_changes_in_set_of_device < args.num_sessions:
                for k, buf in init_warm_global_model_weight.items():
                    # copy_(…) handles device mismatches (GPU→CPU or vice-versa)
                    buf[num_changes_in_set_of_device].copy_(new_global[k])
            else:
                raise IndexError(f"Session index {num_changes_in_set_of_device} out of range [0, {args.num_sessions})")

        global_model.load_state_dict(new_global)
        
        test_dataloader = filter_dataset_by_classes(type_testing_dataset_dict[args.dataset_name], unique_classes, args.batch_size_training, args.dataset_name)

        acc, global_loss = test_inference(args, torch.device(f"cuda:{args.gpu_index}"), global_model, test_dataloader)

        accuracy_list[sync_idx+1] = acc


def maybe_download(url: str, dest: str, retries: int = 3, backoff: float = 5.0):
    """
    Download `url` to `dest` if dest doesn’t exist or is empty.
    Retries up to `retries` times with exponential backoff.
    """
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"[SKIP] {dest} already exists")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"[DOWNLOAD] {url} → {dest} (attempt {attempt}/{retries})")
            urlretrieve(url, dest)
            print(f"[OK] downloaded {dest}")
            return
        except Exception as e:
            print(f"[ERROR] Download failed ({e})")
            if attempt < retries:
                sleep_time = backoff * attempt
                print(f"  retrying in {sleep_time:.1f}s…")
                time.sleep(sleep_time)
            else:
                raise