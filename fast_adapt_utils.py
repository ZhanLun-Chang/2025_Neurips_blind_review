import torch
import random
from fast_adpat_net import net, CNN_SVHN, Res18Con, SVHN_Cons, TextCNN, ResNetFeatureExtractor, MobileNetV2FeatureExtractor, MobileNetV3LargeFeatureExtractor, SVHNFastContrastiveNet, MobileNetV3SmallFeatureExtractor, SVHNEfficient, FastText
import copy
import numpy as np
import json
import torchvision
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torchvision.models as tvmodels
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import os
import torch.nn as nn
import traceback
from typing import Dict, List, Optional, Tuple, Callable, Any
from fast_adapt_dataloader import dict_to_tuple_collate, custom_collate_fn
from collections import Counter
import matplotlib.pyplot as plt
import logging
import torch.optim as optim
from pathlib import Path
from argparse import Namespace
import math
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

    def _cifar10_model(self, algo: str) -> nn.Module:
        
        out_dim = 10
        if algo=='moon':
            return Res18Con(out_dim=256, n_classes=out_dim)
        base = tvmodels.resnet18(weights='DEFAULT')
        base.fc = nn.Linear(base.fc.in_features, out_dim)
        return base

    def _cifar100_model(self, algo: str) -> nn.Module:
        out_dim = 100
        return (
                Res18Con(out_dim=256, n_classes=out_dim)
                if algo=='moon' else
                self._generic_resnet(18, out_dim)
            )

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


def distinct_half(dataset, num_clients):
    class_indices = defaultdict(list)

    # Collect indices for each class
    for idx, data in enumerate(dataset):
        if idx == 0:
            print(type(data[0]))
            print(type(data[1]))
        if isinstance(dataset, (torchvision.datasets.MNIST, torchvision.datasets.FashionMNIST)):
            label = data[1]  # MNIST and Fashion-MNIST
        elif isinstance(dataset, torchvision.datasets.SVHN):
            label = dataset.labels[idx]  # SVHN
        elif isinstance(dataset, (torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100)):
            label = data[1]  # CIFAR-10 and CIFAR-100
        elif isinstance(dataset, ImageFolder):
            label = data[1]
        else:
            raise ValueError("Unsupported dataset type")

        class_indices[label].append(idx)

    # Determine the number of clients and divide them into two groups
    num_classes = len(class_indices)
    half_clients = num_clients // 2
    half_classes = num_classes // 2

    clients_data_ids = {}
    client_classes = {}

    # Assign the first half of the classes to the first half of the clients
    for client_id in range(half_clients):
        client_classes_ids = range(0, half_classes)
        client_indices = []
        for class_id in client_classes_ids:
            num_samples_per_class = len(class_indices[class_id]) // half_clients
            client_indices.extend(class_indices[class_id][:num_samples_per_class])
            # Update class indices after assignment
            class_indices[class_id] = class_indices[class_id][num_samples_per_class:]
        clients_data_ids[client_id] = client_indices
        client_classes[client_id] = list(client_classes_ids)

    # Assign the second half of the classes to the second half of the clients
    for client_id in range(half_clients, num_clients):
        client_classes_ids = range(half_classes, num_classes)
        client_indices = []
        for class_id in client_classes_ids:
            num_samples_per_class = len(class_indices[class_id]) // half_clients
            client_indices.extend(class_indices[class_id][:num_samples_per_class])
            # Update class indices after assignment
            class_indices[class_id] = class_indices[class_id][num_samples_per_class:]
        clients_data_ids[client_id] = client_indices
        client_classes[client_id] = list(client_classes_ids)

    return clients_data_ids, client_classes

def distinct_class_each_device(dataset):

    class_indices = defaultdict(list)
    # Collect indices for each class
    for idx, data in enumerate(dataset):
        if isinstance(dataset, (torchvision.datasets.MNIST, torchvision.datasets.FashionMNIST)):
            label = data[1]  # MNIST and Fashion-MNIST
        elif isinstance(dataset, torchvision.datasets.SVHN):
            label = dataset.labels[idx]  # SVHN
        elif isinstance(dataset, (torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100)):
            label = data[1]  # CIFAR-10 and CIFAR-100
        else:
            raise ValueError("Unsupported dataset type")

        class_indices[label].append(idx)

    # Determine the number of clients and classes per client
    if isinstance(dataset, torchvision.datasets.CIFAR10):
        num_clients = 10
        num_classes_per_client = 1
    elif isinstance(dataset, torchvision.datasets.CIFAR100):
        num_clients = 10
        num_classes_per_client = 10
    elif isinstance(dataset, (torchvision.datasets.MNIST, torchvision.datasets.FashionMNIST)):
        num_clients = 10
        num_classes_per_client = 1
    elif isinstance(dataset, torchvision.datasets.SVHN):
        num_clients = 10
        num_classes_per_client = 1

    clients_data_ids = {}
    client_classes = {}

    for client_id in range(num_clients):
        # Define the classes for each client
        client_classes_ids = range(client_id * num_classes_per_client, (client_id + 1) * num_classes_per_client)
        client_indices = []
        for class_id in client_classes_ids:
            client_indices.extend(class_indices[class_id])
        # Assign the data loader
        clients_data_ids[client_id] = client_indices
        # Store the classes for the current client
        client_classes[client_id] = client_classes_ids

    return clients_data_ids, client_classes


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

def distribute_labels_in_batches(dataset, num_clients):
    """
    Distribute labels in batches for datasets like MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100, and datasets with 200 classes.
    - For 10-label datasets (MNIST, Fashion-MNIST, SVHN, CIFAR-10), each client gets 2 labels.
    - For 100-label datasets (CIFAR-100), each client gets 20 labels.
    - For 200-label datasets, each client gets 20 labels, with each class split into two chunks.
    
    Args:
    - dataset: The dataset to be partitioned.
    - num_clients: The number of clients.

    Returns:
    - clients_data_ids: A dictionary where keys are device IDs, and values are lists of data indices.
    - client_classes: A dictionary where keys are device IDs, and values are sets of assigned labels.
    """
    # Step 1: Extract labels based on dataset type
    if hasattr(dataset, 'targets'):
        # For MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, we convert to numpy for indexing
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        # For SVHN
        targets = np.array(dataset.labels)
    else:
        raise ValueError("Dataset does not have 'targets' or 'labels' attribute")

    # Determine the number of labels in the dataset
    num_classes = len(np.unique(targets))
    
    # Step 2: Group data by label
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        label = targets[idx]
        class_indices[label].append(idx)

    # Shuffle the labels
    available_labels = list(class_indices.keys())
    np.random.shuffle(available_labels)

    # Prepare data structures to store client data and assigned labels
    clients_data_ids = {i: [] for i in range(num_clients)}
    client_classes = {i: set() for i in range(num_clients)}
    client_batch_counts = {i: 0 for i in range(num_clients)}  # To track how many batches each client has been assigned

    # Helper function to assign data and ensure non-empty batches
    def assign_data_to_clients(half_1, half_2, chosen_clients, batch):
        client_1, client_2 = chosen_clients
        if len(half_1) > 0 and len(half_2) > 0:
            clients_data_ids[client_1].extend(half_1)
            clients_data_ids[client_2].extend(half_2)
            client_classes[client_1].update(batch)
            client_classes[client_2].update(batch)
            return True
        return False

    # If dataset has 10 labels (MNIST, Fashion-MNIST, SVHN, CIFAR-10)
    if num_classes == 10:
        # Create 10 batches, each with 1 label
        label_batches = [[label] for label in available_labels]
        
        # Split each batch into halves and assign to two random clients
        for batch in label_batches:
            half_1 = []
            half_2 = []

            for label in batch:
                indices = class_indices[label]
                mid_point = len(indices) // 2
                half_1.extend(indices[:mid_point])
                half_2.extend(indices[mid_point:])

            # Select two random clients that have not already received 2 batches
            available_clients = [client for client, count in client_batch_counts.items() if count < 2]
            chosen_clients = np.random.choice(available_clients, 2, replace=False)

            # Ensure clients get non-empty batches
            if assign_data_to_clients(half_1, half_2, chosen_clients, batch):
                client_batch_counts[chosen_clients[0]] += 1
                client_batch_counts[chosen_clients[1]] += 1

    # If dataset has 100 labels (CIFAR-100)
    elif num_classes == 100:
        num_labels_per_batch = 10
        num_batches = 10

        # Step 3: Create batches of 10 non-overlapping labels
        label_batches = [available_labels[i * num_labels_per_batch: (i + 1) * num_labels_per_batch] for i in range(num_batches)]

        # Step 4: Assign each batch to two randomly selected clients
        for batch in label_batches:
            # Split the data into two halves
            half_1 = []
            half_2 = []

            for label in batch:
                indices = class_indices[label]
                mid_point = len(indices) // 2
                half_1.extend(indices[:mid_point])
                half_2.extend(indices[mid_point:])

            # Select two random clients that have not already received 2 batches
            available_clients = [client for client, count in client_batch_counts.items() if count < 2]
            chosen_clients = np.random.choice(available_clients, 2, replace=False)

            # Ensure clients get non-empty batches
            if assign_data_to_clients(half_1, half_2, chosen_clients, batch):
                client_batch_counts[chosen_clients[0]] += 1
                client_batch_counts[chosen_clients[1]] += 1

    # If dataset has 200 labels (new case)
    elif num_classes == 200:
        # Step 1: Create chunks for each class (two chunks per class)
        class_chunks = {}
        for label, indices in class_indices.items():
            np.random.shuffle(indices)  # Shuffle indices for random chunk division
            mid_point = len(indices) // 2
            class_chunks[label] = [indices[:mid_point], indices[mid_point:]]  # Split into two chunks

        # Step 2: Create a pool of (label, chunk_index) pairs
        chunk_pool = [(label, chunk_index) for label in class_chunks for chunk_index in [0, 1]]
        np.random.shuffle(chunk_pool)  # Shuffle to randomize chunk assignment

        # Track chunks already assigned to clients
        assigned_chunks = defaultdict(set)  # Keep track of which chunk has been assigned to each client

        # Ensure that chunk pool has 400 chunks (2 chunks per class, 200 classes)
        if len(chunk_pool) != 400:
            print(f"Error: The chunk pool has {len(chunk_pool)} chunks instead of 400.")
            raise ValueError("The chunk pool has an unexpected number of chunks.")

        # Step 3: Assign chunks to clients
        for client_id in range(num_clients):
            assigned_labels = set()

            while len(assigned_labels) < 20:
                if len(chunk_pool) == 0:
                    print(f"Error: Not enough chunks left in the pool for client {client_id}.")
                    raise ValueError("Not enough chunks available to assign.")

                # Pick a random chunk from the pool
                selected_label, chunk_index = chunk_pool.pop()

                # Ensure the chunk hasn't been assigned to the client yet
                if selected_label not in client_classes[client_id]:
                    # Assign the chunk to the client
                    clients_data_ids[client_id].extend(class_chunks[selected_label][chunk_index])
                    client_classes[client_id].add(selected_label)
                    assigned_labels.add(selected_label)
                    
                    # Mark this chunk as assigned
                    assigned_chunks[selected_label].add(chunk_index)
                else:
                    # If the chunk has already been assigned, put it back into the pool
                    chunk_pool.append((selected_label, chunk_index))
                    np.random.shuffle(chunk_pool)  # Re-shuffle the pool to maintain randomness
    else:
        raise ValueError(f"Unsupported dataset with {num_classes} labels.")

    return clients_data_ids, client_classes


def distribute_labels_slight_overlap_clients(dataset, total_clients):
    """
    Distributes data indices across clients based on labels, ensuring two sets of labels
    with a specified overlap (approx 20%) are assigned to two halves of the clients.

    Args:
        dataset: PyTorch-style Dataset (e.g., MNIST, CIFAR-100) or HF Dataset.
                 Must allow indexing and have a 'targets' or 'labels' attribute/column.
        total_clients: The number of clients to distribute data across (must be >= 2).

    Returns:
        clients_data_ids: Dict mapping client_id (0 to total_clients-1) to list of data indices.
        client_classes: Dict mapping client_id to set of unique labels assigned.
    """
    if total_clients < 2:
        raise ValueError("total_clients must be at least 2 for this split.")

    # Step 1: Extract all labels efficiently
    targets = None
    if hasattr(dataset, 'targets'): # Common for torchvision datasets
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'): # Common for older torchvision or custom datasets
        targets = np.array(dataset.labels)
    elif 'label' in getattr(dataset, 'features', {}): # Common for Hugging Face datasets
        try:
            # Efficiently get the whole column if possible
            label_data = dataset['label']
            if isinstance(label_data, list):
                 targets = np.array(label_data)
            elif hasattr(label_data, 'numpy'): # Check if it's a tensor
                 targets = label_data.numpy()
            else: # Fallback if it's some other iterable
                 targets = np.array(list(label_data))
        except Exception as e:
            print(f"Warning: Could not access 'label' column efficiently ({e}). Falling back.")
    elif hasattr(dataset, '__getitem__'): # Fallback: Iterate if needed (slower)
        print("Warning: Accessing labels via __getitem__, this might be slow.")
        try:
             # Assuming __getitem__ returns dict or tuple containing label
             first_item = dataset[0]
             if isinstance(first_item, dict) and 'label' in first_item:
                 targets = np.array([dataset[i]['label'] for i in range(len(dataset))])
             elif isinstance(first_item, (tuple, list)) and len(first_item) > 1:
                 targets = np.array([dataset[i][1] for i in range(len(dataset))]) # Assuming label is 2nd elem
             else:
                  raise ValueError("Cannot determine label structure from dataset[0]")
        except Exception as fallback_e:
             raise ValueError(f"Dataset does not have 'targets', 'labels', or accessible 'label' column/item. Error: {fallback_e}")
    else:
         raise ValueError("Cannot extract labels from the provided dataset.")

    # Convert labels to integers if they aren't already
    targets = targets.astype(int)
    unique_labels = np.unique(targets)
    num_labels = len(unique_labels)

    if num_labels < 2:
        raise ValueError(f"Dataset must have at least 2 unique labels for overlap split, found {num_labels}")

    # Step 2: Define label sets (ensure robustness for small num_labels)
    # Aim for 60% per set, 20% overlap relative to total labels
    num_labels_per_set = max(1, int(0.6 * num_labels))
    overlap_size = max(0, min(num_labels_per_set, int(0.2 * num_labels))) # Overlap can't exceed set size

    # Ensure sorted order for consistent selection
    all_sorted_labels = sorted(unique_labels.tolist())

    # Set 1: First num_labels_per_set
    labels_set1 = set(all_sorted_labels[:num_labels_per_set])

    # Overlap: Last 'overlap_size' elements of the chosen labels_set1
    overlap_labels = set(sorted(list(labels_set1))[-overlap_size:]) if overlap_size > 0 else set()

    # Set 2: Needs 'num_labels_per_set' labels total
    # Start with overlap, add remaining needed from labels *not* in set 1
    labels_needed_for_set2 = num_labels_per_set - len(overlap_labels)
    remaining_available_labels = [lbl for lbl in all_sorted_labels if lbl not in labels_set1]
    labels_set2 = overlap_labels.union(set(remaining_available_labels[:labels_needed_for_set2]))

    # Ensure set 2 has the correct size if remaining labels were insufficient
    if len(labels_set2) < num_labels_per_set and len(labels_set1) > len(labels_set2):
         # Borrow from non-overlap part of set 1 if needed
         borrow_needed = num_labels_per_set - len(labels_set2)
         non_overlap_set1_sorted = sorted(list(labels_set1 - overlap_labels))
         labels_set2.update(non_overlap_set1_sorted[:borrow_needed])
         print(f"Warning: Adjusted label_set2 size due to limited unique labels.")

    non_overlap_set1 = labels_set1 - overlap_labels
    non_overlap_set2 = labels_set2 - overlap_labels

    print(f"Num Labels: {num_labels}, Labels per Set: {num_labels_per_set}, Overlap Size: {len(overlap_labels)}")
    print(f"Set 1 ({len(labels_set1)}): {sorted(list(labels_set1))}")
    print(f"Set 2 ({len(labels_set2)}): {sorted(list(labels_set2))}")
    print(f"Overlap ({len(overlap_labels)}): {sorted(list(overlap_labels))}")

    # Step 3: Group indices by class and shuffle within class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    for label in class_indices:
        random.shuffle(class_indices[label]) # Shuffle indices once per class

    # Step 4: Distribute indices to clients
    clients_data_ids = defaultdict(list)
    client_classes = defaultdict(set)
    half_clients = total_clients // 2 # Number of clients in the first group

    # Iterate through each label and decide where its data goes
    all_participating_labels = labels_set1.union(labels_set2)

    for label in all_participating_labels:
        indices_for_label = class_indices[label]
        num_samples_for_label = len(indices_for_label)

        if num_samples_for_label == 0:
            continue # Skip if a label somehow has no samples

        if label in overlap_labels:
            # Split data roughly 50/50, assign first half to Group 1 clients, second to Group 2
            split_point = num_samples_for_label // 2
            first_half_indices = indices_for_label[:split_point]
            second_half_indices = indices_for_label[split_point:]

            # Distribute first half to first 'half_clients'
            if half_clients > 0 and len(first_half_indices) > 0:
                client_splits1 = np.array_split(first_half_indices, half_clients)
                for i in range(half_clients):
                    client_id = i
                    assigned_indices = client_splits1[i].tolist()
                    if assigned_indices: # Only add if list is not empty
                       clients_data_ids[client_id].extend(assigned_indices)
                       client_classes[client_id].add(label)

            # Distribute second half to remaining clients (total_clients - half_clients)
            num_group2_clients = total_clients - half_clients
            if num_group2_clients > 0 and len(second_half_indices) > 0:
                client_splits2 = np.array_split(second_half_indices, num_group2_clients)
                for i in range(num_group2_clients):
                    client_id = half_clients + i
                    assigned_indices = client_splits2[i].tolist()
                    if assigned_indices:
                       clients_data_ids[client_id].extend(assigned_indices)
                       client_classes[client_id].add(label)

        elif label in non_overlap_set1:
            # Assign all data for this label only to Group 1 clients
            if half_clients > 0:
                client_splits = np.array_split(indices_for_label, half_clients)
                for i in range(half_clients):
                    client_id = i
                    assigned_indices = client_splits[i].tolist()
                    if assigned_indices:
                        clients_data_ids[client_id].extend(assigned_indices)
                        client_classes[client_id].add(label)

        elif label in non_overlap_set2:
            # Assign all data for this label only to Group 2 clients
            num_group2_clients = total_clients - half_clients
            if num_group2_clients > 0:
                client_splits = np.array_split(indices_for_label, num_group2_clients)
                for i in range(num_group2_clients):
                    client_id = half_clients + i
                    assigned_indices = client_splits[i].tolist()
                    if assigned_indices:
                        clients_data_ids[client_id].extend(assigned_indices)
                        client_classes[client_id].add(label)

    # Ensure all clients are represented, even if they got no data (unlikely with this setup)
    for i in range(total_clients):
        if i not in clients_data_ids:
            clients_data_ids[i] = []
        if i not in client_classes:
            client_classes[i] = set()

    # Convert defaultdicts back to regular dicts for return (optional)
    return dict(clients_data_ids), dict(client_classes)

def print_client_data_distribution(clients_data_ids, client_classes, dataset):
    """
    Print the labels each client has and how many datapoints for each label.

    Args:
    - clients_data_ids: A dictionary where keys are client IDs, and values are lists of data indices.
    - client_classes: A dictionary where keys are client IDs, and values are sets of assigned labels.
    - dataset: The dataset from which data is being distributed (used to map data indices back to labels).
    """
    
    # Step 1: Extract labels based on dataset type
    if hasattr(dataset, 'targets'):
        # For MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, we convert to numpy for indexing
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        # For SVHN
        targets = np.array(dataset.labels)
    else:
        raise ValueError("Dataset does not have 'targets' or 'labels' attribute")

    # Step 2: Count the number of datapoints for each label per client
    for client_id, data_indices in clients_data_ids.items():
        print(f"\nClient {client_id}:")
        
        # Count the number of datapoints for each label
        label_counts = defaultdict(int)
        for index in data_indices:
            label = targets[index]
            label_counts[label] += 1
        
        # Print the labels and the corresponding number of datapoints
        for label in sorted(client_classes[client_id]):
            print(f"  Label {label}: {label_counts[label]} datapoints")

def distinct_half_agnews(dataset, num_clients):
    """
    Distribute AG_NEWS dataset indices across num_clients:
      - First half get data from most frequent labels (Group 1)
      - Second half get data from less frequent labels (Group 2)

    Args:
        dataset: A Hugging Face Dataset object with 'torch' format
                 and 'label' column (e.g., processed_train).
        num_clients: total number of clients (must be even)

    Returns:
        clients_data_ids: dict mapping client_id to list of dataset indices
        client_classes: dict mapping client_id to list of unique labels
    """
    assert num_clients % 2 == 0, "num_clients must be even"

    # Get all labels efficiently as a tensor, then convert to list
    try:
        # Assumes 'label' column exists and format is torch or numpy-compatible
        label_tensor = dataset['label']
        # Convert tensor to list of ints (move to CPU if necessary)
        if hasattr(label_tensor, 'cpu'): # Check if it's a tensor requiring cpu()
            all_labels = label_tensor.cpu().numpy().astype(int).tolist()
        else: # Assume it might already be numpy array or list
            all_labels = np.array(label_tensor).astype(int).tolist()
    except Exception as e:
        print(f"Error accessing labels efficiently: {e}")
        print("Falling back to slower item-by-item access...")
        # Fallback (slower): iterate item by item accessing by key
        all_labels = [int(dataset[i]['label']) for i in range(len(dataset))]

    label_counter = Counter(all_labels)

    # Check if there are enough unique labels
    if len(label_counter) < 4:
        raise ValueError(f"Expected at least 4 unique labels for AG News split, found {len(label_counter)}")

    # Sort labels by frequency (most to least)
    sorted_labels = sorted(label_counter.keys(), key=lambda l: label_counter[l], reverse=True)

    # Define two groups based on label frequency
    group1_labels = set(sorted_labels[:2])  # Top 2 labels
    group2_labels = set(sorted_labels[2:4]) # Next 2 labels

    group1_indices, group2_indices = [], []

    # Split indices into two groups
    # Use enumerate(all_labels) which is efficient now
    for idx, label in enumerate(all_labels):
        if label in group1_labels:
            group1_indices.append(idx)
        elif label in group2_labels:
            group2_indices.append(idx)
        # Optional: handle cases where label might not be in top 4 if dataset is noisy
        # else:
        #    print(f"Warning: Label {label} at index {idx} not in top 4 frequent labels.")

    print("Total Group 1 samples:", len(group1_indices))
    print("Total Group 2 samples:", len(group2_indices))

    # Shuffle indices
    random.shuffle(group1_indices)
    random.shuffle(group2_indices)

    half_clients = num_clients // 2

    # Check if groups are empty before splitting (can happen with small datasets/splits)
    if not group1_indices or not group2_indices:
        raise ValueError("One or both label groups have zero samples after filtering. Cannot split.")

    # Split indices among clients in each group
    group1_splits = np.array_split(group1_indices, half_clients)
    group2_splits = np.array_split(group2_indices, half_clients)

    clients_data_ids = {}
    client_classes = {}

    # Assign Group 1 clients
    for i in range(half_clients):
        # Ensure split is not empty before converting
        indices = group1_splits[i].tolist() if group1_splits[i].size > 0 else []
        clients_data_ids[i] = indices
        # Get labels corresponding to these specific indices
        client_classes[i] = sorted(list(set(all_labels[idx] for idx in indices))) if indices else []


    # Assign Group 2 clients
    for i in range(half_clients):
        client_id = half_clients + i
        # Ensure split is not empty before converting
        indices = group2_splits[i].tolist() if group2_splits[i].size > 0 else []
        clients_data_ids[client_id] = indices
        # Get labels corresponding to these specific indices
        client_classes[client_id] = sorted(list(set(all_labels[idx] for idx in indices))) if indices else []

    return clients_data_ids, client_classes

def distribute_agnews_labels_specific(dataset, total_clients):
    """
    Distributes AG News data indices using ONLY 3 of its 4 labels across clients:
    - Label 0 (1st sorted unique) -> Group 1 Clients (First Half)
    - Label 1 (2nd sorted unique) -> Group 2 Clients (Second Half)
    - Label 2 (3rd sorted unique) -> All Clients (Shared)
    - Label 3 (4th sorted unique) -> THIS LABEL'S DATA IS NOT USED/DISTRIBUTED.

    Args:
        dataset: Processed AG News Dataset object (e.g., HF Dataset with 'torch' format).
                Must allow indexing and have accessible 'label' information.
        total_clients: The total number of clients (>= 2, ideally even).

    Returns:
        clients_data_ids: Dict mapping client_id (0 to total_clients-1) to list of data indices.
        client_classes: Dict mapping client_id to set of unique labels assigned (will only contain 0, 1, 2).
    """
    # --- Input Validation ---
    if total_clients < 2:
        raise ValueError("total_clients must be at least 2 for this split.")
    if total_clients % 2 != 0:
        print(f"Warning: total_clients ({total_clients}) is odd. Groups will have slightly uneven sizes.")

    # --- Step 1: Extract all labels efficiently ---
    targets = None
    if hasattr(dataset, 'targets'):  # Common for torchvision datasets
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):  # Common for older torchvision or custom datasets
        targets = np.array(dataset.labels)
    elif 'label' in getattr(dataset, 'features', {}):  # Common for Hugging Face datasets
        try:
            label_data = dataset['label']
            if isinstance(label_data, list):
                targets = np.array(label_data)
            elif hasattr(label_data, 'numpy'):  # Check if it's a tensor
                # Move to CPU just in case it's on GPU
                targets = label_data.cpu().numpy() if hasattr(label_data, 'cpu') else label_data.numpy()
            else:  # Fallback if it's some other iterable
                targets = np.array(list(label_data))
        except Exception as e:
            print(f"Warning: Could not access 'label' column efficiently ({e}). Falling back.")
    elif hasattr(dataset, '__getitem__'):  # Fallback: Iterate if needed (slower)
        print("Warning: Accessing labels via __getitem__, this might be slow.")
        try:
            # Assuming __getitem__ returns dict or tuple containing label
            first_item = dataset[0]
            if isinstance(first_item, dict) and 'label' in first_item:
                targets = np.array([dataset[i]['label'] for i in range(len(dataset))])
            elif isinstance(first_item, (tuple, list)) and len(first_item) > 1:
                targets = np.array([dataset[i][1] for i in range(len(dataset))])  # Assuming label is 2nd elem
            else:
                raise ValueError("Cannot determine label structure from dataset[0]")
        except Exception as fallback_e:
            raise ValueError(f"Dataset does not have 'targets', 'labels', or accessible 'label' column/item. Error: {fallback_e}")
    else:
        raise ValueError("Cannot extract labels from the provided dataset.")

    # Convert labels to integers if they aren't already
    targets = targets.astype(int)
    unique_labels = np.unique(targets)
    num_labels = len(unique_labels)

    # --- Step 2: Validate dataset has exactly 4 labels ---
    if num_labels != 4:
        raise ValueError(f"Expected exactly 4 unique labels for AG News specific split, but found {num_labels}.")

    # Use sorted unique labels for consistent assignment
    sorted_unique_labels = sorted(unique_labels.tolist())
    label_grp1_only = sorted_unique_labels[0]  # Will be assigned to Group 1
    label_grp2_only = sorted_unique_labels[1]  # Will be assigned to Group 2
    label_shared = sorted_unique_labels[2]     # Will be assigned to All Clients
    label_unused = sorted_unique_labels[3]     # Will be IGNORED
    print(f"Assigning roles: {label_grp1_only}->Grp1, {label_grp2_only}->Grp2, {label_shared}->All, {label_unused}->UNUSED")

    # --- Step 3: Group indices by class *only for the labels being used* ---
    class_indices = defaultdict(list)
    labels_to_use = {label_grp1_only, label_grp2_only, label_shared}
    for idx, label in enumerate(targets):
        if label in labels_to_use:
            class_indices[label].append(idx)

    # Shuffle only the indices we are using
    for label in class_indices:
        random.shuffle(class_indices[label])

    # --- Step 4: Initialize outputs and define client groups ---
    clients_data_ids = defaultdict(list)
    client_classes = defaultdict(set)
    half_clients = total_clients // 2
    num_group1_clients = half_clients
    # Ensure Group 2 gets the remainder if total_clients is odd
    num_group2_clients = total_clients - half_clients

    group1_client_ids = list(range(num_group1_clients))
    group2_client_ids = list(range(num_group1_clients, total_clients))
    all_client_ids = list(range(total_clients))

    # --- Helper Function for Distribution ---
    def assign_data(label, target_client_ids, indices_to_split):
        """Splits indices for a label and assigns them to target clients."""
        num_target_clients = len(target_client_ids)
        # Only proceed if there are clients and data to assign
        if num_target_clients > 0 and indices_to_split:
            client_splits = np.array_split(indices_to_split, num_target_clients)
            for i, client_id in enumerate(target_client_ids):
                assigned_indices = client_splits[i].tolist()
                # Check if the split is non-empty before assigning
                if assigned_indices:
                    clients_data_ids[client_id].extend(assigned_indices)
                    client_classes[client_id].add(label)
    # --- End Helper Function ---

    # --- Step 5: Perform distribution using only the 3 selected labels ---
    # Label exclusive to Group 1
    if label_grp1_only in class_indices:  # Check if label exists in data
        assign_data(label_grp1_only, group1_client_ids, class_indices[label_grp1_only])

    # Label exclusive to Group 2
    if label_grp2_only in class_indices:
        assign_data(label_grp2_only, group2_client_ids, class_indices[label_grp2_only])

    # Shared label -> All clients
    if label_shared in class_indices:
        assign_data(label_shared, all_client_ids, class_indices[label_shared])

    # *** Data for label_unused is explicitly NOT DISTRIBUTED ***
    print(f"Data for label {label_unused} was intentionally ignored.")

    # --- Step 6: Ensure all clients keys exist in the output dicts ---
    for i in range(total_clients):
        if i not in clients_data_ids:
            clients_data_ids[i] = []
        if i not in client_classes:
            client_classes[i] = set()  # Will be empty if client got no data

    # Convert defaultdicts back to regular dicts
    return dict(clients_data_ids), dict(client_classes)

def distribute_agnews_exclusive_quarters(dataset, total_clients):
    """
    Distributes AG News data indices (4 labels) across clients, assigning
    each label exclusively to one quarter of the clients based on sorted label order.
    - Label 0 (1st sorted) -> Quarter 1 Clients
    - Label 1 (2nd sorted) -> Quarter 2 Clients
    - Label 2 (3rd sorted) -> Quarter 3 Clients
    - Label 3 (4th sorted) -> Quarter 4 Clients

    Args:
        dataset: Processed AG News Dataset object (e.g., HF Dataset with 'torch' format).
                 Must allow indexing and have accessible 'label' information.
        total_clients: The total number of clients (MUST be divisible by 4 and >= 4).

    Returns:
        clients_data_ids: Dict mapping client_id (0 to total_clients-1) to list of data indices.
        client_classes: Dict mapping client_id to set containing the single unique label assigned.
    """
    # --- Input Validation ---
    if total_clients < 4:
        raise ValueError("total_clients must be at least 4 for this split.")
    if total_clients % 4 != 0:
        raise ValueError("total_clients MUST be divisible by 4 for this distribution.")

    # --- Step 1: Extract all labels efficiently ---
    targets = None
    # --- Start copied label extraction ---
    if hasattr(dataset, 'targets'):  # Common for torchvision datasets
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):  # Common for older torchvision or custom datasets
        targets = np.array(dataset.labels)
    elif 'label' in getattr(dataset, 'features', {}):  # Common for Hugging Face datasets
        try:
            label_data = dataset['label']
            if isinstance(label_data, list):
                targets = np.array(label_data)
            elif hasattr(label_data, 'numpy'):  # Check if it's a tensor
                targets = label_data.cpu().numpy() if hasattr(label_data, 'cpu') else label_data.numpy()
            else:
                targets = np.array(list(label_data))
        except Exception as e:
            print(f"Warning: Could not access 'label' column efficiently ({e}). Falling back.")
    elif hasattr(dataset, '__getitem__'):  # Fallback: Iterate if needed (slower)
        print("Warning: Accessing labels via __getitem__, this might be slow.")
        try:
            first_item = dataset[0]
            if isinstance(first_item, dict) and 'label' in first_item:
                targets = np.array([dataset[i]['label'] for i in range(len(dataset))])
            elif isinstance(first_item, (tuple, list)) and len(first_item) > 1:
                targets = np.array([dataset[i][1] for i in range(len(dataset))])
            else:
                raise ValueError("Cannot determine label structure from dataset[0]")
        except Exception as fallback_e:
            raise ValueError(f"Dataset does not have 'targets', 'labels', or accessible 'label' column/item. Error: {fallback_e}")
    else:
        raise ValueError("Cannot extract labels from the provided dataset.")
    # --- End copied label extraction ---

    # Convert labels to integers if they aren't already
    targets = targets.astype(int)
    unique_labels = np.unique(targets)
    num_labels = len(unique_labels)

    # --- Step 2: Validate label count ---
    if num_labels != 4:
        raise ValueError(f"Expected exactly 4 unique labels for AG News 4-quarter split, but found {num_labels}.")

    # Use sorted unique labels for consistent assignment
    sorted_unique_labels = sorted(unique_labels.tolist())
    label_q1 = sorted_unique_labels[0]
    label_q2 = sorted_unique_labels[1]
    label_q3 = sorted_unique_labels[2]
    label_q4 = sorted_unique_labels[3]
    print(f"Assigning roles based on sorted labels: {label_q1}->Q1, {label_q2}->Q2, {label_q3}->Q3, {label_q4}->Q4")

    # --- Step 3: Group indices by class and shuffle ---
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    for label in class_indices:
        random.shuffle(class_indices[label])

    # --- Step 4: Initialize outputs and define client quarters ---
    clients_data_ids = defaultdict(list)
    client_classes = defaultdict(set)
    quarter_size = total_clients // 4

    q1_client_ids = list(range(0 * quarter_size, 1 * quarter_size))
    q2_client_ids = list(range(1 * quarter_size, 2 * quarter_size))
    q3_client_ids = list(range(2 * quarter_size, 3 * quarter_size))
    q4_client_ids = list(range(3 * quarter_size, 4 * quarter_size))
    all_quarter_ids = [q1_client_ids, q2_client_ids, q3_client_ids, q4_client_ids]
    all_quarter_labels = [label_q1, label_q2, label_q3, label_q4]

    # --- Helper Function for Distribution ---
    def assign_data(label, target_client_ids, indices_to_split):
        """Splits indices for a label and assigns them to target clients."""
        num_target_clients = len(target_client_ids)
        if num_target_clients > 0 and indices_to_split:
            client_splits = np.array_split(indices_to_split, num_target_clients)
            for i, client_id in enumerate(target_client_ids):
                assigned_indices = client_splits[i].tolist()
                if assigned_indices:
                    clients_data_ids[client_id].extend(assigned_indices)
                    client_classes[client_id].add(label)
    # --- End Helper Function ---

    # --- Step 5: Perform distribution, assigning each label to its quarter ---
    for i in range(4):
        current_label = all_quarter_labels[i]
        current_quarter_client_ids = all_quarter_ids[i]
        if current_label in class_indices:
            assign_data(current_label, current_quarter_client_ids, class_indices[current_label])
        else:
            print(f"Warning: Label {current_label} expected but not found in dataset indices.")

    # --- Step 6: Ensure all clients keys exist in the output dicts ---
    for i in range(total_clients):
        if i not in clients_data_ids:
            clients_data_ids[i] = []
        if i not in client_classes:
            client_classes[i] = set()

    return dict(clients_data_ids), dict(client_classes)

def pathological_non_iid_partition(dataset, num_clients):
    """
    Partitions a dataset into a pathological non-IID distribution for Federated Learning.
    Each client is assigned data from a limited number of shards, typically resulting
    in each client having data from only a few classes.

    This method sorts data by label, divides it into shards, and assigns a fixed
    number of shards (specifically 2, based on the described paper's method) to each client.

    Args:
        dataset: The dataset to be partitioned. This can be a dataset object
                 with 'targets' or 'labels' attribute (like torchvision datasets)
                 or a dataset object from the 'datasets' library with a 'label' column.
        num_clients: The number of clients to partition the data among.

    Returns:
        clients_data_ids: A dictionary where keys are client IDs (0 to num_clients-1)
                          and values are lists of data indices assigned to each client.
        client_classes: A dictionary where keys are client IDs (0 to num_clients-1)
                        and values are sets of unique assigned class labels for each client.

    Raises:
        ValueError: If the dataset does not have a recognizable label attribute or column,
                    or if the total number of shards (num_clients * 2) is not
                    divisible by the number of classes, or if the total number of
                    samples is not divisible by the total number of shards.
    """
    # Step 1: Extract labels based on dataset type
    targets = None
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    elif 'label' in dataset.features:
        print("Detected 'label' column in dataset features. Extracting labels...")
        label_column = dataset['label']
        if isinstance(label_column, np.ndarray):
            targets = label_column.astype(int)
        elif hasattr(label_column, 'tolist'):
            targets = np.array(label_column).astype(int)
        elif hasattr(label_column, 'cpu') and hasattr(label_column, 'numpy'):
            targets = label_column.cpu().numpy().astype(int)
        elif hasattr(label_column, 'numpy'):
            targets = label_column.numpy().astype(int)
        else:
            raise TypeError(f"Could not convert dataset['label'] of type {type(label_column)} to numpy array.")
    else:
        raise ValueError("Dataset does not have a recognizable label attribute ('targets', 'labels') or a 'label' column in its features.")

    total_samples = len(dataset)
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)

    # Step 2: Group data indices by label
    index_to_label = {i: targets[i] for i in range(total_samples)}
    class_indices = defaultdict(list)
    for idx in range(total_samples):
        label = index_to_label[idx]
        class_indices[label].append(idx)
    class_indices = {label: class_indices[label] for label in sorted(class_indices.keys())}

    # Step 3: Calculate shard parameters
    shards_per_client = 2
    total_shards = num_clients * shards_per_client
    if total_shards % num_classes != 0:
        raise ValueError(
            f"Total number of shards ({total_shards}) must be divisible by "
            f"the number of classes ({num_classes}) for equal distribution per label."
        )
    shards_per_label = total_shards // num_classes
    if total_samples % total_shards != 0:
        raise ValueError(
            f"Total number of samples ({total_samples}) must be divisible by "
            f"the total number of shards ({total_shards}) for equal shard sizes."
        )
    shard_size = total_samples // total_shards

    # Step 4: Create the shards
    all_shards = []
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        np.random.shuffle(indices)
        for i in range(shards_per_label):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size
            all_shards.append(indices[start_idx:end_idx])

    # Step 5: Shuffle the shards
    np.random.shuffle(all_shards)

    # Step 6: Assign shards to clients
    clients_data_ids = {i: [] for i in range(num_clients)}
    client_classes = {i: set() for i in range(num_clients)}
    shard_idx = 0
    for client_id in range(num_clients):
        assigned_shards = all_shards[shard_idx: shard_idx + shards_per_client]
        shard_idx += shards_per_client
        for shard in assigned_shards:
            clients_data_ids[client_id].extend(shard)
            if len(shard) > 0:
                first_index_in_shard = shard[0]
                label_of_shard = index_to_label[first_index_in_shard]
                client_classes[client_id].add(label_of_shard)

    return clients_data_ids, client_classes

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
    shards_per_client: int = 2,
    unbalanced_sgm: float = 0.0
):
    """
    Partitions a dataset for federated learning into two client-groups (A, B),
    splits the label space into two subsets (with optional overlap between groups for different sessions),
    allocates samples to clients within a session/group based on the specified distribution method, and builds:
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

    In-session label distribution methods (`in_session_label_dist`):
      (Determines how samples of the group-specific labels are distributed among clients *within* that group)
      - "dirichlet":
          For each client group (A or B) and its assigned set of labels:
            1. Determine total samples per client: The total number of data samples
               available for the group's labels is distributed among the clients in
               that group. If `unbalanced_sgm > 0`, this distribution is drawn
               from a log-normal distribution (controlled by `unbalanced_sgm` as sigma),
               ensuring some clients get more data than others. Otherwise, data is
               distributed as evenly as possible. This determines each client's "quota".
            2. Client-specific class proportions: For each client, a set of proportions
               for the labels in its group's label set is drawn from a
               Dirichlet distribution (parameterized by `dirichlet_alpha`).
            3. Sample assignment: Samples are assigned to clients iteratively. For each
               sample slot in a client's quota, a class is chosen based on the
               client's Dirichlet-generated class proportions. If samples of that
               class are available in the (shuffled) pool for the group's labels,
               one is assigned. If prior-based selection fails after several attempts
               (e.g., preferred classes are exhausted), a fallback mechanism assigns
               a sample from any other available class in the group to fill the slot.
               This continues until client quotas are met or data runs out for the group.
      - "two_shards":
          For each label-group:
            • total_shards = shards_per_client * group_size
            • shards_per_label = floor(total_shards / num_labels)
            • For each label: split its indices into `shards_per_label` shards
              of size floor(len(idxs)/shards_per_label), leave remainder unused.
            • Pool all shards, shuffle, then for each client draw
              `shards_per_client` shards without replacement.
            • Each client hence gets at most `shards_per_client` labels.

    Args:
        dataset:
            Any dataset with integer labels accessible as described in `_extract_targets`.
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
            Fraction of label overlap between Group A's label set and Group B's label set.
            If 0: disjoint label halves; if in (0,1): attempt that fraction overlap.
        in_session_label_dist (str):
            Method for distributing samples of group-specific labels to clients within that group.
            Options: "dirichlet" or "two_shards".
        dirichlet_alpha (float):
            α parameter for Dirichlet when `in_session_label_dist` is "dirichlet".
        shards_per_client (int):
            # of shards per client when `in_session_label_dist` is "two_shards".
        unbalanced_sgm (float):
            Sigma for log-normal distribution of samples per client when `in_session_label_dist`
            is "dirichlet". If 0, samples are distributed as evenly as possible.

    Returns:
        clients_data_ids (dict[int, list[int]]):
            client_id → assigned sample indices.
        client_classes (dict[int, set[int]]):
            client_id → set of labels held.
        session_clients (list[list[int]]):
            length = num_sessions; client-group per session.
        training_order (list[list[int]]):
            per-round client IDs; total rounds =
            num_sessions_pilot*num_rounds_pilot +
            (num_sessions-num_sessions_pilot)*num_rounds_actual

    Raises:
        ValueError:
          - If `num_clients_per_group < 1`.
          - If `num_sessions_pilot > num_sessions`.
          - If `cross_session_label_overlap` not in [0,1).
          - If `in_session_label_dist` invalid.
    """
    # --- Argument Validation ---
    # Ensure the number of clients per group is at least 1.
    if num_clients_per_group < 1:
        raise ValueError("num_clients_per_group must be ≥ 1.")
    # Ensure pilot sessions do not exceed total sessions.
    if num_sessions_pilot > num_sessions:
        raise ValueError("num_sessions_pilot cannot exceed num_sessions.")
    # Ensure overlap fraction is within the valid range [0, 1).
    if not (0 <= cross_session_label_overlap < 1):
        raise ValueError("cross_session_label_overlap must be in [0,1).")
    # Ensure a valid allocation method is specified.
    if in_session_label_dist not in ("dirichlet", "two_shards"):
        raise ValueError("in_session_label_dist must be 'dirichlet' or 'two_shards'.")
    # Ensure sigma for log-normal distribution is non-negative.
    if unbalanced_sgm < 0:
        raise ValueError("unbalanced_sgm must be non-negative.")

    # --- 1. Extract Labels from Dataset ---
    targets = _extract_targets(dataset) # Extract all target labels using the helper function.
    unique_labels = np.unique(targets)  # Get the sorted list of unique labels present in the dataset.
    L = unique_labels.size              # Total number of unique classes in the dataset (L).

    # --- 2. Define Client Groups ---
    total_clients = 2 * num_clients_per_group # Calculate the total number of clients.
    clients = list(range(total_clients))      # Create a list of client IDs (0 to total_clients - 1).
    groupA = clients[:num_clients_per_group]  # Assign the first half of clients to Group A.
    groupB = clients[num_clients_per_group:]  # Assign the second half of clients to Group B.

    # --- 3. Split Labels for Group A and Group B ---
    # This section determines which labels are primarily associated with Group A and Group B across different sessions.
    # The `cross_session_label_overlap` parameter controls how many labels are shared between these two primary sets.
    if L == 0: # Handle the case of an empty dataset (no labels).
        labels1 = [] # Labels for Group A will be empty.
        labels2 = [] # Labels for Group B will be empty.
    elif cross_session_label_overlap == 0: # No overlap: Split labels into two disjoint halves.
        half = L // 2 # Integer division to find the midpoint.
        labels1 = unique_labels[:half].tolist() # Group A gets the first half of unique labels.
        labels2 = unique_labels[half:].tolist() # Group B gets the second half.
    else: # Calculate overlapping label sets based on `cross_session_label_overlap`.
        # G: Effective size of a single group's label pool if labels were distributed to fill (2 - overlap_fraction) conceptual slots.
        # O: Number of overlapping labels between the two groups.
        denominator = (2 - cross_session_label_overlap)
        if denominator == 0: G = L # Should ideally not happen due to overlap_fraction < 1.
        else: G = int(np.floor(L / denominator))
        O = int(np.floor(G * cross_session_label_overlap))

        # If calculated overlap (O) is 0, or group size (G) is 0, or overlap is greater than or equal to group size,
        # it's not a valid overlap scenario as defined. Fallback to a simple disjoint split.
        if O == 0 or G == 0 or G <= O : 
            half = L // 2
            labels1 = unique_labels[:half].tolist()
            labels2 = unique_labels[half:].tolist()
        else:
            U = G - O # Number of unique (non-overlapping) labels for each group's initial part.
            
            # Define slices for unique and overlapping parts, ensuring indices are within bounds of `unique_labels`.
            uniq1_end = U
            overl_start = U
            overl_end = U + O
            uniq2_start = U + O 
            uniq2_end = U + O + U 

            # Extract label arrays for unique parts and the overlapping part.
            uniq1_labels = unique_labels[:min(uniq1_end, L)]
            overl_labels = unique_labels[min(overl_start, L):min(overl_end, L)]
            uniq2_labels = unique_labels[min(uniq2_start, L):min(uniq2_end, L)] # These are unique to the "second" conceptual group.
            
            # Construct label sets for Group A (labels1) and Group B (labels2).
            labels1_list = []
            if uniq1_labels.size > 0: labels1_list.extend(uniq1_labels.tolist())
            if overl_labels.size > 0: labels1_list.extend(overl_labels.tolist())
            
            labels2_list = []
            if overl_labels.size > 0: labels2_list.extend(overl_labels.tolist()) 
            if uniq2_labels.size > 0: labels2_list.extend(uniq2_labels.tolist())
            
            # Ensure labels are unique within each list (in case of edge conditions in slicing) and sorted.
            labels1 = sorted(list(set(labels1_list)))
            labels2 = sorted(list(set(labels2_list)))
    
    # Debug print to show the resulting label sets for each group.
    print(f"[DEBUG] Labels for Group A (labels1): {labels1} (Count: {len(labels1)})")
    print(f"[DEBUG] Labels for Group B (labels2): {labels2} (Count: {len(labels2)})")


    # --- 4. Allocate Data Samples to Clients ---
    # Initialize dictionaries to store data sample indices and class sets for each client.
    clients_data_ids = {cid: [] for cid in clients} # Maps client_id to a list of sample_indices.
    client_classes   = {cid: set() for cid in clients}  # Maps client_id to a set of labels they hold.

    # --- Method 1: Dirichlet-based distribution ---
    if in_session_label_dist == "dirichlet":
        # This loop processes Group A with labels1, then Group B with labels2.
        for lbls_for_group, grp_client_ids in ((labels1, groupA), (labels2, groupB)):
            num_clients_in_grp = len(grp_client_ids)
            # Skip if the current group is empty or has no labels assigned to it.
            if num_clients_in_grp == 0 or not lbls_for_group: 
                continue

            # Collect all sample indices for the labels assigned to this current group.
            group_label_indices = {} # Maps label -> list of sample_indices for that label within this group.
            total_samples_for_group_labels = 0
            for lbl in lbls_for_group:
                idxs = np.where(targets == lbl)[0] # Find all samples in the entire dataset with the current label.
                group_label_indices[lbl] = idxs
                total_samples_for_group_labels += len(idxs)
            
            # If no samples exist for any of the group's assigned labels, skip processing this group.
            if total_samples_for_group_labels == 0:
                continue 

            # --- 4.1. Determine client quotas (target number of samples per client in this group) ---
            client_target_quotas_list = np.zeros(num_clients_in_grp, dtype=int) # Initialize quotas to zero.
            if total_samples_for_group_labels > 0 : # Proceed only if there's data to distribute.
                avg_samples_per_client_in_grp = total_samples_for_group_labels / num_clients_in_grp
                
                # If unbalanced distribution is requested (`unbalanced_sgm > 0`) and possible.
                if unbalanced_sgm > 0 and num_clients_in_grp > 0: 
                    # Use log-normal distribution to create varying quotas for clients.
                    # `mu` is the mean of the underlying normal distribution for the log-normal.
                    mu = np.log(avg_samples_per_client_in_grp) if avg_samples_per_client_in_grp > 1e-9 else -10 # Avoid log(0).
                    quotas_raw = np.random.lognormal(mean=mu, sigma=unbalanced_sgm, size=num_clients_in_grp)
                    
                    sum_quotas_raw = np.sum(quotas_raw)
                    if sum_quotas_raw > 1e-9: # Avoid division by zero.
                        # Normalize raw quotas so their sum matches the total available samples for the group.
                        client_target_quotas_list_float = (quotas_raw / sum_quotas_raw) * total_samples_for_group_labels
                    else: # Fallback if lognormal gives all zeros (e.g., due to extreme sigma).
                        client_target_quotas_list_float = np.full(num_clients_in_grp, avg_samples_per_client_in_grp)
                else: # Balanced distribution: each client gets roughly an equal number of samples.
                    client_target_quotas_list_float = np.full(num_clients_in_grp, avg_samples_per_client_in_grp)

                client_target_quotas_list = client_target_quotas_list_float.astype(int) # Convert float quotas to integer counts.
                
                # Adjust for rounding errors to ensure the sum of integer quotas matches total available samples.
                current_sum_quotas = np.sum(client_target_quotas_list)
                diff_sum = total_samples_for_group_labels - current_sum_quotas
                
                if diff_sum != 0: # If there's a difference due to rounding.
                    # Distribute the remainder (or deficit) one by one among clients cyclically.
                    for i in range(int(abs(diff_sum))):
                        client_target_quotas_list[i % num_clients_in_grp] += np.sign(diff_sum)
                
                client_target_quotas_list = np.maximum(0, client_target_quotas_list) # Ensure no client has a negative quota.
                
                # Final check and correction if the sum is still off (e.g., if all quotas became 0 then diff_sum was positive).
                final_sum_check = np.sum(client_target_quotas_list)
                if final_sum_check != total_samples_for_group_labels:
                    if final_sum_check == 0 and total_samples_for_group_labels > 0:
                        # If all quotas are zero but samples exist, distribute them (e.g., to the first few clients).
                        temp_total_to_distribute = total_samples_for_group_labels
                        for i in range(num_clients_in_grp):
                            can_take = temp_total_to_distribute 
                            client_target_quotas_list[i] = can_take
                            temp_total_to_distribute -= can_take
                            if temp_total_to_distribute == 0: break
                    elif final_sum_check > 0 : # If sum is non-zero but still off, perform one more adjustment pass.
                        diff_again = total_samples_for_group_labels - final_sum_check
                        for i in range(int(abs(diff_again))):
                             client_target_quotas_list[i % num_clients_in_grp] += np.sign(diff_again)
                        client_target_quotas_list = np.maximum(0, client_target_quotas_list)

            # --- 4.2. Client-specific Class Priors (Proportions for labels in lbls_for_group) ---
            client_prior_cumsum_map = {} # Stores cumulative sum of Dirichlet-generated priors for each client.
            if lbls_for_group: # Only if there are labels to distribute for this group.
                for cid in grp_client_ids:
                    # Each client gets its own set of label proportions drawn from a Dirichlet distribution.
                    # `dirichlet_alpha` controls the uniformity of these proportions.
                    priors = np.random.dirichlet(alpha=[dirichlet_alpha] * len(lbls_for_group))
                    client_prior_cumsum_map[cid] = np.cumsum(priors) # Store cumulative sum for efficient sampling later.
            
            # --- 4.3. Create Data Pools (available samples for each label in the group, shuffled) ---
            # `indices_by_label_pool` maps each label to a list of its available sample indices.
            indices_by_label_pool = {lbl: list(group_label_indices[lbl]) for lbl in lbls_for_group}
            for lbl in indices_by_label_pool:
                np.random.shuffle(indices_by_label_pool[lbl]) # Shuffle samples within each label's pool for randomness.
            
            # `samples_left_in_label_pool` tracks the count of remaining samples for each label.
            samples_left_in_label_pool = {lbl: len(indices_by_label_pool[lbl]) for lbl in lbls_for_group}

            # --- 4.4. Iterative Sample Assignment to clients in the current group ---
            for client_grp_idx, current_client_id in enumerate(grp_client_ids):
                num_samples_this_client_needs = client_target_quotas_list[client_grp_idx] # Get the quota for this client.
                
                # Skip if client needs 0 samples, or no labels/priors defined for the group.
                if num_samples_this_client_needs == 0 or not lbls_for_group or not client_prior_cumsum_map:
                    continue
                
                client_specific_prior_cumsum = client_prior_cumsum_map[current_client_id] # Get client's label preferences.

                # Fill the client's quota slot by slot.
                for _slot_idx in range(num_samples_this_client_needs):
                    # If no data is left in any label pool for this group, stop assigning to this client.
                    if sum(samples_left_in_label_pool.values()) == 0:
                        break 

                    assigned_this_slot = False
                    # Try to pick a class based on the client's Dirichlet priors (with multiple attempts).
                    # Max attempts is a heuristic to give a fair chance for prior-based selection.
                    for _attempt in range(len(lbls_for_group) * 3 + 5): 
                        u_draw = np.random.uniform() # Random draw for inverse transform sampling.
                        # Select a label based on the client's prior distribution using the cumulative sum.
                        chosen_label_idx_in_lbls = np.argmax(u_draw <= client_specific_prior_cumsum)
                        chosen_label = lbls_for_group[chosen_label_idx_in_lbls]

                        # If samples of the chosen label are available, assign one.
                        if samples_left_in_label_pool[chosen_label] > 0:
                            sample_to_assign = indices_by_label_pool[chosen_label].pop() # Get a sample index from the pool.
                            clients_data_ids[current_client_id].append(sample_to_assign) # Assign to client.
                            client_classes[current_client_id].add(chosen_label) # Record that client has this label.
                            samples_left_in_label_pool[chosen_label] -= 1 # Decrement count for that label's pool.
                            assigned_this_slot = True
                            break # Slot filled, move to the next slot for this client.

                    if not assigned_this_slot:
                        # Fallback: If prior-based selection failed (e.g., preferred classes exhausted for this client's prior),
                        # pick any available label from the group's remaining pool to fill the slot.
                        available_labels_for_fallback = [
                            lbl for lbl in lbls_for_group if samples_left_in_label_pool[lbl] > 0
                        ]
                        if available_labels_for_fallback:
                            fallback_label = np.random.choice(available_labels_for_fallback) # Randomly pick from available labels.
                            sample_to_assign = indices_by_label_pool[fallback_label].pop()
                            clients_data_ids[current_client_id].append(sample_to_assign)
                            client_classes[current_client_id].add(fallback_label)
                            samples_left_in_label_pool[fallback_label] -= 1
                
                # If all data for the group is exhausted, no need to process further clients in this group.
                if sum(samples_left_in_label_pool.values()) == 0:
                    break 
    
    # --- Method 2: Shard-based distribution ("two_shards") ---
    elif in_session_label_dist == "two_shards":
        # This loop processes Group A with labels1, then Group B with labels2.
        for group_idx, (lbls, grp) in enumerate(((labels1, groupA), (labels2, groupB))):
            M = len(grp) # Number of clients in the current group.
            # Skip if group is empty, clients aren't configured to take shards, or no labels for this group.
            if M == 0 or shards_per_client == 0 or not lbls:
                continue
            
            # Total number of shards to be distributed among clients in this group.
            total_shards_for_group = shards_per_client * M
            num_labels_for_group = len(lbls) # Number of unique labels assigned to this group.

            if num_labels_for_group == 0: # Should be caught by `not lbls` already.
                continue 
            
            all_shards_created_for_group = [] # List to hold all shards created from all labels in this group.
            
            # Only proceed if we are supposed to make any shards at all for this group.
            if total_shards_for_group > 0:
                # Calculate nominal number of shards to create per label.
                shards_per_label_nominal = total_shards_for_group // num_labels_for_group

                if shards_per_label_nominal >= 1:
                    # Standard case: create `shards_per_label_nominal` shards for each label.
                    for lbl_idx, lbl in enumerate(lbls):
                        idxs_for_label = np.where(targets == lbl)[0] # Get all samples for the current label.
                        if len(idxs_for_label) == 0: continue # Skip if label has no samples.
                        np.random.shuffle(idxs_for_label) # Shuffle samples of the current label.
                        
                        # Calculate the size of each shard for this specific label.
                        calculated_shard_size = len(idxs_for_label) // shards_per_label_nominal
                        
                        if calculated_shard_size == 0: 
                            # If a label has too few samples for `shards_per_label_nominal` shards of size > 0,
                            # but `shards_per_label_nominal` is >=1 (meaning we *should* make shards for this label),
                            # create one shard with all its (non-empty) samples.
                            if len(idxs_for_label) > 0:
                                all_shards_created_for_group.append(idxs_for_label.tolist())
                            continue # Move to next label.
                        
                        # Create `shards_per_label_nominal` shards of `calculated_shard_size`.
                        current_sample_idx_ptr = 0
                        shards_made_for_this_label = 0
                        for _ in range(shards_per_label_nominal):
                            start_slice = current_sample_idx_ptr
                            end_slice = start_slice + calculated_shard_size
                            if end_slice <= len(idxs_for_label): # Check if enough samples remain for a full shard.
                                shard_content = idxs_for_label[start_slice:end_slice].tolist()
                                if shard_content: # Ensure shard is not empty.
                                    all_shards_created_for_group.append(shard_content)
                                    shards_made_for_this_label +=1
                                current_sample_idx_ptr = end_slice # Advance pointer.
                            else: break # Not enough for another full shard from this label; remainder is unused.
                else: 
                    # Fallback case: `shards_per_label_nominal` is 0, but `total_shards_for_group` > 0.
                    # This means num_labels_for_group > total_shards_for_group.
                    # We need to create `total_shards_for_group` shards in total.
                    # Iterate through labels (shuffled for fairness) and create one shard 
                    # (containing all samples for that label) from each until `total_shards_for_group` are made.
                    shards_created_count = 0
                    shuffled_lbls_for_group = np.random.permutation(lbls).tolist() # Shuffle labels to pick fairly.

                    for lbl_idx, lbl in enumerate(shuffled_lbls_for_group):
                        if shards_created_count >= total_shards_for_group:
                            break # Enough shards have been created for the group.

                        idxs_for_label = np.where(targets == lbl)[0]
                        if len(idxs_for_label) > 0: # Only create a shard if the label has samples.
                            all_shards_created_for_group.append(idxs_for_label.tolist()) # Shard contains all samples for this label.
                            shards_created_count += 1
            # If total_shards_for_group is 0, all_shards_created_for_group remains empty.
            
            # Shuffle all collected shards from all labels in this group before assigning to clients.
            np.random.shuffle(all_shards_created_for_group) 
            
            # Assign up to `shards_per_client` shards to each client in the group.
            current_shard_idx_ptr = 0
            for cid in grp: # For each client in the current group.
                num_shards_assigned_to_client = 0
                # Assign shards while client needs more and shards are available from the pool.
                while num_shards_assigned_to_client < shards_per_client and \
                      current_shard_idx_ptr < len(all_shards_created_for_group):
                    
                    shard_to_assign = all_shards_created_for_group[current_shard_idx_ptr]
                    if shard_to_assign: # Ensure shard is not empty before assigning.
                        clients_data_ids[cid].extend(shard_to_assign)
                        # Determine the label of this shard (all samples in a shard have the same original label).
                        label_of_this_shard = targets[shard_to_assign[0]] 
                        client_classes[cid].add(label_of_this_shard) # Record label for the client.
                        num_shards_assigned_to_client += 1
                    current_shard_idx_ptr += 1 # Move to the next available shard.

    # --- 5. Define Sessions and Training Order ---
    # Create a list of client groups for each session, alternating between Group A and Group B.
    session_clients = [
        groupA.copy() if (s % 2 == 0) else groupB.copy() # Group A for even-numbered sessions, Group B for odd.
        for s in range(num_sessions)
    ]
    
    # Expand the session schedule into a per-round training order.
    training_order = [] # List of lists, where each inner list is the client IDs for a round.
    for s, sess_grp in enumerate(session_clients):
        # Determine number of rounds for this session (pilot phase or actual phase).
        reps = num_rounds_pilot if s < num_sessions_pilot else num_rounds_actual
        if sess_grp: # Ensure the session group is not empty.
            # Add the client group `reps` times to the training order, meaning this group trains for `reps` rounds.
            training_order += [sess_grp.copy()] * reps

    return clients_data_ids, client_classes, session_clients, training_order

def visualize_client_data_distribution(
    clients_data_ids: dict,
    dataset: any,
    output_filename: str = "client_data_distribution.png"
):
    """
    Visualizes the distribution of data samples per class for each client
    as a stacked horizontal bar chart, with no extra padding above/below.
    """
    if not clients_data_ids:
        print("Client data IDs dictionary is empty. Nothing to visualize.")
        return

    # --- extract targets from dataset (you must have your _extract_targets defined) ---
    try:
        dataset_targets = _extract_targets(dataset)
    except ValueError as e:
        print(f"Error extracting targets from dataset: {e}")
        return

    if not isinstance(dataset_targets, np.ndarray):
        print(f"Error: _extract_targets did not return a NumPy array. Got {type(dataset_targets)}.")
        return

    # handle empty-dataset edge
    if dataset_targets.size == 0:
        if len(dataset) == 0:
            print("Dataset is empty. Nothing to visualize.")
        else:
            print("Warning: extracted targets empty but dataset non-empty.")
        return

    # --- infer number of clients ---
    max_cid = max([cid for cid in clients_data_ids.keys() if isinstance(cid, int) and cid >= 0], default=-1)
    num_clients = max_cid + 1
    if num_clients <= 0:
        print("No valid client IDs found. Nothing to visualize.")
        return

    # --- infer number of classes ---
    unique_labels = np.unique(dataset_targets)
    if unique_labels.size == 0:
        print("No labels found. Nothing to plot.")
        return
    num_classes = int(unique_labels.max()) + 1

    print(f"Inferred num_clients: {num_clients}, num_classes: {num_classes}")

    # --- build count matrix ---
    client_class_counts = np.zeros((num_clients, num_classes), dtype=int)
    for cid, indices in clients_data_ids.items():
        if not isinstance(cid, int) or cid < 0 or cid >= num_clients:
            print(f"Skipping invalid client ID {cid}")
            continue
        valid = [i for i in indices if 0 <= i < len(dataset_targets)]
        if not valid:
            continue
        labels = dataset_targets[valid]
        lbs, cnts = np.unique(labels, return_counts=True)
        for lb, ct in zip(lbs, cnts):
            client_class_counts[cid, int(lb)] = ct

    # --- set up figure size (just big enough for the bars) ---
    bar_h = 0.3  # inches per client
    fig_h = np.clip(num_clients * bar_h, 4, 30)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # --- choose colors ---
    if num_classes <= 10:
        cmap = plt.cm.get_cmap('tab10', num_classes)
        colors = [cmap(i) for i in range(num_classes)]
    elif num_classes <= 20:
        cmap = plt.cm.get_cmap('tab20', num_classes)
        colors = [cmap(i) for i in range(num_classes)]
    else:
        cmap = plt.cm.get_cmap('nipy_spectral', num_classes)
        colors = [cmap(i/(num_classes-1)) for i in range(num_classes)]

    # --- plot stacked bars ---
    y_labels = [f"Client {i}" for i in range(num_clients)]
    left = np.zeros(num_clients, dtype=int)
    for cls in range(num_classes):
        counts = client_class_counts[:, cls]
        ax.barh(y_labels, counts, left=left, color=colors[cls], label=f"Class {cls}")
        left += counts

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Client")

    # --- legend ---
    if num_classes <= 50:
        cols = 1 + (num_classes > 20) + (num_classes > 40)
        fontsize = 'medium' if num_classes <= 15 else 'small'
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                  fontsize=fontsize, ncol=cols)

    # ── HERE IS THE MAGIC TO REMOVE TOP/BOTTOM PADDING ──
    ax.set_ylim(-0.5, num_clients - 0.5)
    plt.tight_layout(pad=0.1)
    # ───────────────────────────────────────────────────────

    # --- save ---
    outdir = os.path.dirname(output_filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_filename}")


def report_client_data_summary(
    clients_data_ids: dict,
    dataset: any
):
    """
    Reports the data distribution for each client.
    For each client, it details:
    - Total number of data samples.
    - The set of unique labels they possess.
    - For each of those labels, the count of data samples.

    Args:
        clients_data_ids (dict[int, list[int]]):
            A dictionary where keys are client IDs and values are lists of
            sample indices assigned to that client.
        dataset (any):
            The original dataset object. The function will attempt to extract
            target labels from this object.
    """
    if not clients_data_ids:
        print("Client data IDs dictionary is empty. No summary to report.")
        return

    try:
        dataset_targets = _extract_targets(dataset)
    except ValueError as e:
        print(f"Error extracting targets from dataset: {e}")
        return

    if dataset_targets.size == 0:
        print("Extracted dataset targets are empty. Cannot generate summary.")
        return

    print("--- Client Data Summary ---")
    
    # Determine the full set of client IDs to iterate through, even if some have no data
    # This ensures all clients up to the max ID are considered if clients_data_ids is sparse.
    max_client_id = -1
    if clients_data_ids: # Check if dictionary is not empty
        valid_cids = [cid for cid in clients_data_ids.keys() if isinstance(cid, int) and cid >=0]
        if valid_cids:
            max_client_id = max(valid_cids)
    
    num_clients_to_report = max_client_id + 1 if max_client_id != -1 else 0
    if num_clients_to_report == 0 and clients_data_ids: # Handle cases where keys might not be 0-indexed ints
        print("Warning: No valid non-negative integer client IDs found. Reporting based on dictionary keys.")
        client_id_iterator = sorted(clients_data_ids.keys())
    else:
        client_id_iterator = range(num_clients_to_report)


    for client_id in client_id_iterator:
        print(f"\nClient ID: {client_id}")
        
        if client_id not in clients_data_ids or not clients_data_ids[client_id]:
            print("  Total Samples: 0")
            print("  Labels Held: None")
            continue

        sample_indices_for_client = clients_data_ids[client_id]
        total_samples_for_client = len(sample_indices_for_client)
        print(f"  Total Samples: {total_samples_for_client}")

        if total_samples_for_client == 0:
            print("  Labels Held: None")
            continue
            
        # Ensure indices are valid for dataset_targets
        valid_indices = [idx for idx in sample_indices_for_client if idx < len(dataset_targets)]
        if len(valid_indices) != total_samples_for_client:
            print(f"  Warning: {total_samples_for_client - len(valid_indices)} sample indices were out of bounds "
                  f"for dataset_targets (length {len(dataset_targets)}) and were ignored.")
        
        if not valid_indices:
            print("  Labels Held (after filtering invalid indices): None")
            continue

        labels_for_client_samples = dataset_targets[valid_indices]
        
        unique_labels, counts = np.unique(labels_for_client_samples, return_counts=True)
        
        if unique_labels.size == 0:
            print("  Labels Held: None (no valid labels found for this client's samples)")
        else:
            print(f"  Labels Held (Count):")
            for label, count in zip(unique_labels, counts):
                print(f"    - Label {label}: {count} samples")
    
    print("\n--- End of Summary ---")

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

def get_learning_rate_from_config(args: Namespace) -> float:
    """
    Load and return the learning rate for a given dataset and algorithm
    from a JSON file.

    Expects:
      - args.lr_config_path (str or Path)
      - args.dataset_name (str)
      - args.algorithm (str)

    Raises:
        FileNotFoundError: if the config file is missing.
        ValueError: if required args are missing or keys not found.
        json.JSONDecodeError: if the JSON is invalid.
        TypeError: if the retrieved learning rate is not a number.
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

    # Validate type
    if not isinstance(lr_value, (int, float)):
        raise TypeError(
            f"Expected numeric learning rate for {ds_name}/{alg_name}; "
            f"got {type(lr_value).__name__}."
        )

    return float(lr_value)

def get_init_lr_step_gamma_from_config(args: Namespace) -> Tuple[float, int, float]:
    """
    Load and return (init_lr, step_size, gamma) for a given dataset and algorithm
    from a JSON file where each entry is [init_lr, step_size, gamma].

    Expects:
      - args.lr_config_path (str or Path)
      - args.dataset_name (str)
      - args.algorithm (str)

    Raises:
        FileNotFoundError: config file is missing.
        ValueError: invalid or missing arguments, or keys not found.
        json.JSONDecodeError: invalid JSON format.
        TypeError: wrong-shaped entry or wrong types in config.
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
    params = ds_cfg.get(alg_name)
    if params is None:
        available = ", ".join(ds_cfg.keys())
        raise ValueError(
            f"Algorithm '{alg_name}' not defined for dataset '{ds_name}' "
            f"(available: {available})."
        )

    # Validate shape
    if not isinstance(params, (list, tuple)) or len(params) != 3:
        raise TypeError(
            f"Expected a list/tuple of three values [init_lr, step_size, gamma] for "
            f"{ds_name}/{alg_name}; got {params!r}."
        )

    # Unpack and validate types
    init_lr, step_size, gamma = params
    if not isinstance(init_lr, (int, float)):
        raise TypeError(f"Expected numeric init_lr for {ds_name}/{alg_name}; got {type(init_lr).__name__}.")
    if not isinstance(step_size, int):
        raise TypeError(f"Expected integer step_size for {ds_name}/{alg_name}; got {type(step_size).__name__}.")
    if not isinstance(gamma, (int, float)):
        raise TypeError(f"Expected numeric gamma for {ds_name}/{alg_name}; got {type(gamma).__name__}.")

    return float(init_lr), step_size, float(gamma)

def require_StepLR_stepsize(args: Namespace) -> int:
    """
    Fetch args.StepLR_stepsize or fail with a clear message.
    """
    try:
        stepsize = args.StepLR_stepsize
    except AttributeError:
        raise AttributeError(
            "Missing `args.StepLR_stepsize`—make sure you called get_lr_step_gamma_from_config(args) before."
        )
    if not isinstance(stepsize, int):
        raise TypeError(f"Expected args.StepLR_stepsize to be an int, got {type(stepsize).__name__}")
    return stepsize


def require_StepLR_gamma(args: Namespace) -> float:
    """
    Fetch args.StepLR_gamma or fail with a clear message.
    """
    try:
        gamma = args.StepLR_gamma
    except AttributeError:
        raise AttributeError(
            "Missing `args.StepLR_gamma`—make sure you called get_lr_step_gamma_from_config(args) before."
        )
    if not isinstance(gamma, (int, float)):
        raise TypeError(f"Expected args.StepLR_gamma to be a float, got {type(gamma).__name__}")
    return float(gamma)

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
    

def calculate_cosine_annealing_lr_schedule(
    initial_client_lr: float,
    total_communication_rounds: int,
    min_client_lr: float = 0.0 # Minimum learning rate during the cycle
) -> List[float]:
    """
    Calculates the learning rates for each communication round following
    a Cosine Annealing schedule.

    Args:
        initial_client_lr: The initial learning rate for clients
                           (the maximum LR in the cycle).
        total_communication_rounds: The total number of communication rounds
                                    over which the learning rate will decay
                                    from initial_client_lr to min_client_lr.
        min_client_lr: The minimum learning rate at the end of the cosine cycle.

    Returns:
        A list of learning rates, where the i-th element is the learning rate
        for the i-th communication round (starting from round 0).
        The length of the list is total_communication_rounds.
    """
    lr_values = []
    eta_max = initial_client_lr
    eta_min = min_client_lr
    t_max = total_communication_rounds

    for t in range(t_max):
        # Cosine Annealing formula:
        # lr_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / t_max))
        lr_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / t_max))
        lr_values.append(lr_t)

    return lr_values

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

def calculate_step_lr_schedule(
    initial_client_lr: float,
    total_communication_rounds: int,
    step_size: int,
    gamma: float = 0.1,
    min_client_lr: float = 0.0
) -> List[float]:
    """
    Calculates a step-decay learning rate schedule:
        lr(t) = max(initial_client_lr * gamma^(floor(t / step_size)), min_client_lr)

    Args:
        initial_client_lr: The starting (peak) learning rate.
        total_communication_rounds: Total number of rounds over which to schedule.
        step_size: Number of rounds between each decay step.
        gamma: Multiplicative factor of learning rate decay (0 < gamma < 1).
        min_client_lr: The floor learning rate at the end of the schedule.

    Returns:
        A list of length `total_communication_rounds` where
        each element is the LR for that round (rounds indexed from 0 to total_communication_rounds-1).
    """
    lr_values: List[float] = []
    for t in range(total_communication_rounds):
        # how many steps have occurred by round t
        num_steps = t // step_size
        lr_t = initial_client_lr * (gamma ** num_steps)
        # enforce the minimum learning rate
        if lr_t < min_client_lr:
            lr_t = min_client_lr
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

        for _ in range(args.footprint_num_iteration):

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