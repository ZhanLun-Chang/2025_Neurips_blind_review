from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import zipfile
import requests
from PIL import Image
import random
import pandas as pd
from collections import Counter, defaultdict
import torch
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset, concatenate_datasets
import json
from typing import List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.utils import download_url




tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

class PretrainedTokenizerProcessor:
    def __init__(self, model_name="bert-base-uncased", max_len=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.pad_idx = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.vocab_size = self.tokenizer.vocab_size

    def tokenize_and_pad(self, examples): # Use plural for batched=True
        # Tokenizer handles batching efficiently
        encoded = self.tokenizer(
            examples["text"], # Process batch of texts
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        # Return dict compatible with dataset columns
        return {
            "input_ids": encoded["input_ids"],
            # Ensure labels are ints before set_format("torch") needs them
            "label": [int(lbl) for lbl in examples["label"]]
        }

    def apply(self, dataset):
        # Use batched=True for significant speedup
        processed_dataset = dataset.map(self.tokenize_and_pad, batched=True)
        # Set format to TORCH
        processed_dataset.set_format(type="torch", columns=["input_ids", "label"])
        # Remove original columns if desired (optional)
        # processed_dataset = processed_dataset.remove_columns(["text"])
        return processed_dataset
    
def custom_collate_fn(list_of_samples):
    """
    Collates a list of samples (each a dict {'pixel_values': tensor, 'label': tensor})
    into a batch tuple (batched_pixel_values, batched_labels).
    """
    # `default_collate` will turn the list of dicts into a dict of batched tensors:
    # e.g., {'pixel_values': batched_pixel_values_tensor, 'label': batched_labels_tensor}
    collated_batch_dict = default_collate(list_of_samples)

    # Extract the tensors and return them as a tuple
    # Ensure the keys 'pixel_values' and 'label' match what your dataset provides
    pixel_values = collated_batch_dict['pixel_values']
    labels = collated_batch_dict['label']
    return pixel_values, labels


def dict_to_tuple_collate(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([])

    inputs_list = [item['input_ids'] for item in batch]
    labels_list = [item['label'] for item in batch]

    inputs_collated = torch.stack(inputs_list, dim=0)
    labels_collated = torch.stack(labels_list, dim=0)

    return inputs_collated, labels_collated

def load_leaf_json(json_path: str):
    """
    Load data from a LEAF JSON file.

    Args:
        json_path (str): Path to the LEAF JSON file.

    Returns:
        The parsed JSON data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an error reading or parsing the JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"LEAF data file not found: {json_path}")

    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise IOError(f"Error reading LEAF JSON file {json_path}: {e}")

def build_vocab_from_leaf(leaf_data):
    """Builds character vocabulary from all users' data."""
    all_chars = set();
    for user_id, user_content in leaf_data['user_data'].items():
        user_text = "".join(user_content.get('x', [])); all_chars.update(user_text)
    if not all_chars: raise ValueError("No chars found in LEAF data.")
    chars = sorted(list(all_chars)); char_to_idx = {ch: i for i, ch in enumerate(chars)}; idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char

def transform_leaf_data_to_sequences(leaf_data, char_to_idx, sequence_length, speaker_to_int_id):
    """
    Transforms loaded LEAF JSON data into a flat list of (input, target)
    pairs and a map from SPEAKER INT ID to data indices.

    For next-character prediction:
        - If the input string (x_str) is longer than sequence_length:
              * Input tensor: first sequence_length tokens.
              * Target tensor: the token at position [sequence_length] (from x_str).
        - If the input string is exactly sequence_length or shorter:
              * Input tensor: the string padded (if necessary) to sequence_length.
              * Target tensor: the provided target character from y.

    Args:
        leaf_data (dict): Dict loaded from LEAF JSON.
        char_to_idx (dict): Mapping from character to integer index.
        sequence_length (int): Desired length of input sequences.
        speaker_to_int_id (dict): Mapping from speaker name (str) to unique integer ID.

    Returns:
        tuple: (D, user_int_id_index_map)
            - D (list): List of tuples: (input_tensor, target_tensor).
              * input_tensor: torch.long, shape=[sequence_length].
              * target_tensor: torch.long, a scalar (if using the provided target) 
                or a scalar from the input string if the string is longer.
            - user_int_id_index_map (dict): Maps speaker INT IDs to lists of indices in D.
    """
    D_list = []
    user_int_id_index_map = {}
    current_index = 0
    unknown_char_index = char_to_idx.get('<UNK>', -1)
    pad_idx = char_to_idx.get('<PAD>', 0)

    print("Transforming LEAF data into (input_tensor, target_tensor) pairs...")

    users_list = leaf_data.get('users', [])
    user_data_dict = leaf_data.get('user_data', {})

    for user_id in users_list:
        if user_id not in user_data_dict:
            continue

        speaker_int_id = speaker_to_int_id.get(user_id)
        if speaker_int_id is None:
            print(f"Warning: Speaker '{user_id}' not in speaker_to_int_id map. Skipping.")
            continue

        user_int_id_index_map.setdefault(speaker_int_id, [])

        user_data = user_data_dict[user_id]
        x_list = user_data.get('x', [])
        y_list = user_data.get('y', [])

        if len(x_list) != len(y_list):
            print(f"Warning: Mismatch between x and y lengths for user '{user_id}'. Skipping.")
            continue
        if not x_list:
            continue

        for sample_idx in range(len(x_list)):
            x_str = x_list[sample_idx]
            y_char = y_list[sample_idx]

            # Process x: convert string to indices.
            x_indices = [char_to_idx.get(c, unknown_char_index) for c in x_str]

            # Decide how to form input and target based on the length of x_indices.
            if len(x_indices) > sequence_length:
                # If longer than desired length, take first sequence_length as input,
                # and use the next token (position sequence_length) as the target.
                input_indices = x_indices[:sequence_length]
                target_idx = x_indices[sequence_length]
            elif len(x_indices) == sequence_length:
                # Use the entire x_indices as input; target from provided y_char.
                input_indices = x_indices
                target_idx = char_to_idx.get(y_char, unknown_char_index)
            else:  # len(x_indices) < sequence_length
                # Pad the input to the desired sequence length.
                input_indices = x_indices + [pad_idx] * (sequence_length - len(x_indices))
                # Use the provided y_char as the target.
                target_idx = char_to_idx.get(y_char, unknown_char_index)

            x_tensor = torch.tensor(input_indices, dtype=torch.long)
            y_tensor = torch.tensor(target_idx, dtype=torch.long)

            D_list.append((x_tensor, y_tensor))
            user_int_id_index_map[speaker_int_id].append(current_index)
            current_index += 1

    print(f"Transformation complete. Total pairs created: {len(D_list)}")
    print(f"User index map created for {len(user_int_id_index_map)} users.")
    return D_list, user_int_id_index_map

def prepare_shakespeare_data(vocab_path="vocab_shakespeare.json"):
    """
    Prepares the Shakespeare dataset for a classification task.
    
    Steps:
      1. Loads the dataset from Hugging Face.
      2. Computes label frequencies and splits the data into two groups:
            - Group 1: Samples with labels among the top 10 most frequent labels.
            - Group 2: Samples with labels among the 11th-20th most frequent labels.
      3. Downsamples the larger group so that its total number of samples is 
         roughly equal to the number of samples in the smaller group. This is done in a stratified
         manner so that each label is proportionally reduced.
      4. For each group, splits the data into 90% train and 10% test.
      5. Combines the splits from both groups.
      6. Builds:
            - An input vocabulary from the training set's 'x' field (character-level).
            - A label vocabulary from the training set's 'y' field.
         If the vocabularies are already saved at `vocab_path`, they are loaded instead.
            
    Returns:
        train_ds: Combined training dataset.
        test_ds: Combined test dataset.
        input_vocab: Dictionary mapping input characters to indices.
        label_vocab: Dictionary mapping label characters to indices.
    """
    # 1. Load the dataset from Hugging Face.
    ds = load_dataset("flwrlabs/shakespeare")
    
    # 2. Compute label frequencies over ds['train'].
    label_counter = Counter(ex['y'] for ex in ds['train'])
    # Sort labels by frequency in descending order.
    sorted_labels = sorted(label_counter.keys(), key=lambda ch: label_counter[ch], reverse=True)
    
    # Define groups based on frequency:
    # Group 1: Top 10 most frequent labels.
    # Group 2: The next 10 most frequent labels (11th-20th).
    group1_label_set = set(sorted_labels[:1])
    group2_label_set = set(sorted_labels[1:3])
    
    # Filter the dataset for each group.
    group1_ds = ds['train'].filter(lambda ex: ex['y'] in group1_label_set)
    group2_ds = ds['train'].filter(lambda ex: ex['y'] in group2_label_set)
    
    print("Group 1 size (before downsampling):", len(group1_ds))
    print("Group 2 size (before downsampling):", len(group2_ds))
    
    # 3. Downsample the larger group so that its size is roughly equal to the smaller group.
    if len(group1_ds) > len(group2_ds):
        desired_size = len(group2_ds)
        current_size = len(group1_ds)
        ratio = desired_size / current_size
        # Build a mapping from label to list of indices for Group 1.
        group1_indices_by_label = {}
        for i, ex in enumerate(group1_ds):
            label = ex['y']
            group1_indices_by_label.setdefault(label, []).append(i)
        
        new_group1_indices = []
        for label, indices in group1_indices_by_label.items():
            count = len(indices)
            new_count = max(1, int(round(count * ratio)))
            if new_count < count:
                sampled = random.sample(indices, new_count)
            else:
                sampled = indices
            new_group1_indices.extend(sampled)
        
        group1_ds = group1_ds.select(new_group1_indices)
        print("Group 1 size (after downsampling):", len(group1_ds))
    elif len(group2_ds) > len(group1_ds):
        desired_size = len(group1_ds)
        current_size = len(group2_ds)
        ratio = desired_size / current_size
        group2_indices_by_label = {}
        for i, ex in enumerate(group2_ds):
            label = ex['y']
            group2_indices_by_label.setdefault(label, []).append(i)
        
        new_group2_indices = []
        for label, indices in group2_indices_by_label.items():
            count = len(indices)
            new_count = max(1, int(round(count * ratio)))
            if new_count < count:
                sampled = random.sample(indices, new_count)
            else:
                sampled = indices
            new_group2_indices.extend(sampled)
        
        group2_ds = group2_ds.select(new_group2_indices)
        print("Group 2 size (after downsampling):", len(group2_ds))
    else:
        print("No downsampling required; both groups are equal in size.")
    
    # 4. For each group, split into 90% train and 10% test.
    group1_split = group1_ds.train_test_split(test_size=0.1, seed=42)
    group2_split = group2_ds.train_test_split(test_size=0.1, seed=42)
    
    # 5. Combine training splits and test splits from both groups.
    train_ds = concatenate_datasets([group1_split['train'], group2_split['train']])
    test_ds = concatenate_datasets([group1_split['test'], group2_split['test']])
    
    print("Combined train set size:", len(train_ds))
    print("Combined test set size:", len(test_ds))
    
    # 6. Build vocabularies (or load if already saved).
    if os.path.exists(vocab_path):
        print(f"Loading vocabularies from {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            input_vocab = vocab_data["input_vocab"]
            label_vocab = vocab_data["label_vocab"]
    else:
        print("Building vocabularies from training data...")
        # Build input vocabulary from the 'x' field in train_ds.
        all_chars = set()
        for ex in train_ds:
            all_chars.update(list(ex['x']))
        all_chars = sorted(list(all_chars))
    
        # Reserve tokens for padding and unknown characters.
        input_vocab = {'<pad>': 0, '<unk>': 1}
        for idx, ch in enumerate(all_chars, start=2):
            input_vocab[ch] = idx
    
        # Build label vocabulary from the 'y' field in train_ds.
        all_labels = sorted(list(set(ex['y'] for ex in train_ds)))
        label_vocab = {label: idx for idx, label in enumerate(all_labels)}
        
        # Ensure the directory exists along the vocab_path.
        vocab_dir = os.path.dirname(vocab_path)
        if vocab_dir and not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir, exist_ok=True)
        
        # Save the vocabularies.
        with open(vocab_path, 'w') as f:
            json.dump({"input_vocab": input_vocab, "label_vocab": label_vocab}, f)
        print(f"Vocabularies saved to {vocab_path}")
    
    return train_ds, test_ds, input_vocab, label_vocab

class ShakespeareDataset(Dataset):
    def __init__(self, hf_dataset, input_vocab, label_vocab, max_length=100):
        """
        hf_dataset: Hugging Face dataset (train or test split)
        input_vocab: dictionary mapping input characters to integers
        label_vocab: dictionary mapping labels (characters) to integers
        max_length: maximum sequence length for input 'x'
        """
        self.data = hf_dataset
        self.input_vocab = input_vocab
        self.label_vocab = label_vocab
        self.max_length = max_length
        
        # Build a labels attribute: a list of integer labels for every sample.
        self.labels = [self.label_vocab[ex['y']] for ex in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        x_str = ex['x']
        y_char = ex['y']

        # Convert input string to list of indices, with truncation/padding.
        x_indices = [self.input_vocab.get(ch, self.input_vocab['<unk>']) for ch in x_str][:self.max_length]
        if len(x_indices) < self.max_length:
            x_indices += [self.input_vocab['<pad>']] * (self.max_length - len(x_indices))
        x_tensor = torch.tensor(x_indices, dtype=torch.long)

        # Convert label character to integer.
        label = self.label_vocab[y_char]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return x_tensor, label_tensor


def build_vocab(texts, vocab_size=5000):
    """
    Build a vocabulary from the provided texts using NLTK's TweetTokenizer.
    Two special tokens are reserved: <PAD> and <UNK>.
    """
    counter = Counter()
    for text in texts:
        tokens = tokenizer.tokenize(text)
        counter.update(tokens)
    most_common = counter.most_common(vocab_size - 2)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    return word2idx

def text_to_sequence(text, word2idx, max_len=40):
    """
    Tokenize the text using LEAF-style TweetTokenizer and convert tokens to indices.
    Pads (or truncates) the sequence to a fixed length.
    """
    tokens = tokenizer.tokenize(text)
    seq = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    if len(seq) < max_len:
        seq += [word2idx['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


class FlatTransformedDataset(Dataset):
    """
    A PyTorch Dataset for the flat list D containing processed
    (input_tensor, target_index_tensor) pairs.

    Args:
        D (list): A flat list where each element is a tuple:
                  (input_sequence_tensor, target_index_tensor).
    """
    def __init__(self, D):
        if not isinstance(D, list):
             raise TypeError(f"Input D must be a list, got {type(D)}")
        self.data_pairs = D
        self.length = len(self.data_pairs)

    def __len__(self):
        """Returns the total number of sequence pairs."""
        return self.length

    def __getitem__(self, idx):
        """Returns the idx-th (input_sequence_tensor, target_index_tensor) pair."""
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.length}")
        # Data should already be tensors
        input_seq_tensor, target_idx_tensor = self.data_pairs[idx]
        # CrossEntropyLoss usually expects target index, not one-hot.
        # Return target as Long tensor, potentially scalar if loss fn handles it.
        # Squeezing target tensor if it has shape [1] to make it scalar-like
        return input_seq_tensor, target_idx_tensor.squeeze() # Squeeze target index

# ------------------------------
# Custom Dataset for Sentiment140
# ------------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=40):
        self.texts = texts
        self.labels = labels  # Already mapped: 0 for negative, 1 for positive.
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = text_to_sequence(self.texts[idx], self.word2idx, self.max_len)
        label = self.labels[idx]  # No need to remap.
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def sentiment140(data_path, vocab_size=5000, max_len=40, train=True, 
                 train_sample_count=720000, test_sample_count=4800):
    """
    Load and preprocess the Sentiment140 dataset.
    
    Expected CSV columns: target, ids, date, flag, user, text.
    Filters out neutral samples, builds the vocabulary, and splits the data into disjoint
    training and testing sets for negative and positive sentiment.
    
    For training:
      - From the training split, sample `train_sample_count` negative and 
        `train_sample_count` positive examples (total = 2 * train_sample_count).
    
    For testing:
      - From the testing split, sample `test_sample_count` negative and 
        `test_sample_count` positive examples (total = 2 * test_sample_count).
    
    Args:
        data_path: Path to the Sentiment140 CSV file.
        vocab_size: Maximum vocabulary size.
        max_len: Maximum token sequence length.
        train: Boolean flag. If True, returns the training split; otherwise, returns the test split.
        train_sample_count: Number of negative (and positive) samples to use for training.
        test_sample_count: Number of negative (and positive) samples to use for testing.
    
    Returns:
        sentiment_dataset: A SentimentDataset instance corresponding to the chosen split.
    """

    # Load the CSV file.
    cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(data_path, names=cols, encoding='latin-1')
    
    # Keep only negative (0) and positive (4) samples.
    df = df[df['target'].isin([0, 4])]
    
    # Build vocabulary from all texts.
    all_texts = df['text'].tolist()
    word2idx = build_vocab(all_texts, vocab_size)
    print(f"Vocabulary size: {len(word2idx)}")
    
    # Split data for negative and positive classes.
    df_neg = df[df['target'] == 0]
    df_pos = df[df['target'] == 4]
    
    # First, split each class into disjoint training and testing sets.
    test_size_ratio = test_sample_count / (train_sample_count + test_sample_count)
    
    df_neg_train, df_neg_test = train_test_split(df_neg, test_size=test_size_ratio, 
                                                  random_state=42, shuffle=True)
    df_pos_train, df_pos_test = train_test_split(df_pos, test_size=test_size_ratio, 
                                                  random_state=42, shuffle=True)
    
    # Sample the desired number of examples.
    if train:
        df_neg_sample = df_neg_train.sample(n=train_sample_count, random_state=42)
        df_pos_sample = df_pos_train.sample(n=train_sample_count, random_state=42)
    else:
        df_neg_sample = df_neg_test.sample(n=test_sample_count, random_state=42)
        df_pos_sample = df_pos_test.sample(n=test_sample_count, random_state=42)
    
    # Concatenate negative and positive samples.
    df_split = pd.concat([df_neg_sample, df_pos_sample])
    
    # Map the target column: negative (0) remains 0; positive (4) becomes 1.
    df_split['target'] = df_split['target'].apply(lambda x: 0 if x == 0 else 1)
    
    # Create the SentimentDataset.
    sentiment_dataset = SentimentDataset(
        df_split['text'].tolist(),
        df_split['target'].tolist(),
        word2idx,
        max_len=max_len
    )
    
    return sentiment_dataset

# Custom class for training data using ImageFolder
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        # keep targets in sync
        self.targets = [label for _, label in self.samples]

    def __getitem__(self, index):
        # if given a batch of indices, return a list of samples
        if isinstance(index, (list, tuple, np.ndarray)):
            return [self._get_item(i) for i in index]
        # else single-sample
        return self._get_item(index)

    def _get_item(self, index):
        path, label = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class TinyImageNetValidationDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load annotations → self.samples (list of image paths) & self.targets
        self.samples, self.targets = self._load_annotations(annotations_file)

    def _load_annotations(self, annotations_file):
        label_map = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                img_name, cls = line.strip().split('\t')[:2]
                label_map[img_name] = cls

        classes = sorted(set(label_map.values()))
        class_to_idx = {c: i for i, c in enumerate(classes)}

        samples, targets = [], []
        for img_name, cls in label_map.items():
            path = os.path.join(self.root_dir, img_name)
            if os.path.exists(path):
                samples.append(path)
                targets.append(class_to_idx[cls])

        return samples, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # handle a batch of indices
        if isinstance(index, (list, tuple, np.ndarray)):
            return [self._get_item(i) for i in index]
        return self._get_item(index)

    def _get_item(self, index):
        img_path = self.samples[index]
        label = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to download and extract dataset if not present
def download_and_extract_tiny_imagenet(data_dir, tiny_imagenet_data_dir):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')

    if not os.path.exists(tiny_imagenet_data_dir):
        os.makedirs(data_dir, exist_ok=True)

        print("Downloading Tiny ImageNet dataset...")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()  # Raises an HTTPError if the response was unsuccessful
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            if downloaded != total_size:
                raise ValueError("Download incomplete: expected {} bytes but got {} bytes".format(total_size, downloaded))

            print("Download complete. Extracting...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

            print("Extraction complete.")
        except Exception as e:
            print(f"Failed to download or extract dataset: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)  # Remove possibly corrupted file
            raise e
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)  # Clean up ZIP file after extraction
    else:
        print("Tiny ImageNet dataset already exists.")


def get_dataset_dict(args, train =True):

    dataroot = os.path.join(os.getcwd(),  "data")
    dataset_name_to_dataset = {}
        
    if args.dataset_name == 'sentiment140':

        sentiment140_csv_file_path = os.path.join(os.getcwd(), "data", "sentiment140", "training.1600000.processed.noemoticon.csv")

        sentiment140_dataset = sentiment140(sentiment140_csv_file_path, vocab_size=5000, max_len=40, train=train, train_sample_count=120000, test_sample_count=60000)
        
        dataset_name_to_dataset[args.dataset_name] = sentiment140_dataset

    elif args.dataset_name == "mnist":
        mnist_dataset = datasets.MNIST(dataroot, train=train,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]),
                            download = True)
        dataset_name_to_dataset[args.dataset_name] = mnist_dataset

    elif args.dataset_name == "fmnist":

        fmnist_dataset = datasets.FashionMNIST(dataroot, train=train,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.2861,),
                                                           (0.3530,))]),
                                    download = True)
        
        dataset_name_to_dataset[args.dataset_name] = fmnist_dataset
    
    elif args.dataset_name == "SVHN":

        if train:
            SVHN_split_str = "train"
        else:
            SVHN_split_str = "test"
        
        SVHN_dataset = datasets.SVHN(dataroot, split = SVHN_split_str,
                                transform=transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    #  transforms.Grayscale(num_output_channels=3),
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                                download = True)

        dataset_name_to_dataset[args.dataset_name] = SVHN_dataset

    elif args.dataset_name == "cifar10":

        # Normalization stats for CIFAR-10
        normalize_cifar10 = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

        # Transformations for the training set (with data augmentation)
        cifar10_train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),      # Randomly crop the image with padding
            transforms.RandomHorizontalFlip(),         # Randomly flip the image horizontally
            transforms.ToTensor(),                     # Convert images to tensor
            normalize_cifar10,                         # Normalize
        ])

        # Transformations for the test set (no augmentation)
        cifar10_test_transform = transforms.Compose([
            transforms.ToTensor(),     # Convert images to tensor
            normalize_cifar10,         # Normalize
        ])

        print(f"Loading CIFAR-10 dataset...")

        if train:
            # Load the training dataset with augmentation
            cifar10_train_dataset = datasets.CIFAR10(root=dataroot, train=True,
                                                    download=True, transform=cifar10_train_transform)
            
            dataset_name_to_dataset[args.dataset_name] = cifar10_train_dataset

            print(f"Finished loading CIFAR-10. Train size: {len(cifar10_train_dataset)}")

        else:

            # Load the test dataset without augmentation
            cifar10_test_dataset = datasets.CIFAR10(root=dataroot, train=False,
                                                    download=True, transform=cifar10_test_transform)
            
            dataset_name_to_dataset[args.dataset_name] = cifar10_test_dataset

            print(f"Finished loading CIFAR-10. Train size: {len(cifar10_test_dataset)}")


        print("Data augmentation applied to the CIFAR-10 training set.")
        

    elif args.dataset_name == "cifar100":

        # Normalization stats for CIFAR-100
        normalize_cifar100 = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])

        # Transformations for the training set (with data augmentation)
        cifar100_train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),      # Randomly crop the image with padding
            transforms.RandomHorizontalFlip(),         # Randomly flip the image horizontally
            transforms.ToTensor(),                     # Convert images to tensor
            normalize_cifar100,                        # Normalize
        ])

        # Transformations for the test set (no augmentation)
        cifar100_test_transform = transforms.Compose([
            transforms.ToTensor(),     # Convert images to tensor
            normalize_cifar100,        # Normalize
        ])

        print(f"Loading CIFAR-100 dataset...")

        if train:
            # Load the training dataset with augmentation
            cifar100_train_dataset = datasets.CIFAR100(root=dataroot, train=True,
                                                    download=True, transform=cifar100_train_transform)
            
            dataset_name_to_dataset[args.dataset_name] = cifar100_train_dataset

            print(f"Finished loading CIFAR-100. Train size: {len(cifar100_train_dataset)}")

        else:
            # Load the test dataset without augmentation
            cifar100_test_dataset = datasets.CIFAR100(root=dataroot, train=False,
                                                    download=True, transform=cifar100_test_transform)
            
            dataset_name_to_dataset[args.dataset_name] = cifar100_test_dataset

            print(f"Finished loading CIFAR-100. Test size: {len(cifar100_test_dataset)}")

        print("Data augmentation applied to the CIFAR-100 training set.")

    
    elif args.dataset_name == "tinyimagenet":

        # # Define data directory
        # data_dir = os.path.join(os.getcwd(), 'data')
        # tiny_imagenet_data_dir = os.path.join(data_dir, 'tiny-imagenet-200')
        

        # # Download dataset if not already downloaded
        # download_and_extract_tiny_imagenet(data_dir, tiny_imagenet_data_dir)

        # if train:
        #     # For Training
        #     train_dir = os.path.join(tiny_imagenet_data_dir, 'train')
        #     train_transforms = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomRotation(10),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ])

        #     dataset_name_to_dataset[args.dataset_name] = CustomImageFolder(root=train_dir, transform=train_transforms)
        # else:
        #     test_dir = os.path.join(tiny_imagenet_data_dir, 'val/images')
        #     annotations_file = os.path.join(tiny_imagenet_data_dir, 'val/val_annotations.txt')

        #     # For Testing
        #     test_transforms = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ])

        #     dataset_name_to_dataset[args.dataset_name] = TinyImageNetValidationDataset(root_dir=test_dir, annotations_file=annotations_file, transform=test_transforms)

        # 1) Download & unzip (one-time)
        data_root = os.path.join(os.getcwd(), "data")
        zip_path  = os.path.join(data_root, "tiny-imagenet-200.zip")
        folder    = os.path.join(data_root, "tiny-imagenet-200")

        os.makedirs(data_root, exist_ok=True)
        if not os.path.isdir(folder):
            print(f"Downloading Tiny-ImageNet to {zip_path} …")
            download_url(
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
                data_root,
                filename="tiny-imagenet-200.zip",
                md5="90528d7ca1a48142e341f4ef8d21d0de"
            )
            print("Extracting …")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_root)
        else:
            print("Found existing Tiny-ImageNet folder, skipping download.")

        # 2) (Optional) Reorganize val/ if you need ImageFolder
        #    The official val/ has all images flat under val/images/
        #    plus val/val_annotations.txt mapping each filename to its class.
        #    If you don’t reorganize, you’ll need a custom Dataset.
        #
        #    Here’s a quick reorg (only do it once):
        val_dir    = os.path.join(folder, "val")
        images_dir = os.path.join(val_dir, "images")
        anno_file  = os.path.join(val_dir, "val_annotations.txt")
        reorg_done = os.path.join(val_dir, ".reorg_done")

        if not os.path.exists(reorg_done):
            print("Reorganizing val split into subfolders …")
            with open(anno_file) as f:
                for line in f:
                    img, cls, *rest = line.strip().split()
                    src = os.path.join(images_dir, img)
                    dst_dir = os.path.join(val_dir, cls)
                    os.makedirs(dst_dir, exist_ok=True)
                    os.rename(src, os.path.join(dst_dir, img))
            # remove the now-empty images/ folder
            os.rmdir(images_dir)
            open(reorg_done, 'w').close()

        # 3) Define your transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # 4) Create ImageFolder datasets
        train_folder = os.path.join(folder, "train")
        val_folder   = os.path.join(folder, "val")

        train_ds = datasets.ImageFolder(train_folder, transform=train_transform)
        val_ds   = datasets.ImageFolder(val_folder,   transform=val_transform)

        if train:
            dataset_name_to_dataset[args.dataset_name] = train_ds
        else:
            dataset_name_to_dataset[args.dataset_name] = val_ds

    else:
        raise ValueError("For the datasetset, we only accpet mnist, fmnist, SVHN, cifar10, cifar100 and tinyimagenet")
    
    return dataset_name_to_dataset

    


