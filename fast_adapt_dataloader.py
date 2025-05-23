from torchvision import datasets, transforms
import os
import zipfile
import torch
from transformers import AutoTokenizer
from torchvision.datasets.utils import download_url


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
    

def dict_to_tuple_collate(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([])

    inputs_list = [item['input_ids'] for item in batch]
    labels_list = [item['label'] for item in batch]

    inputs_collated = torch.stack(inputs_list, dim=0)
    labels_collated = torch.stack(labels_list, dim=0)

    return inputs_collated, labels_collated


def get_dataset_dict(args, train =True):

    dataroot = os.path.join(os.getcwd(),  "data")
    dataset_name_to_dataset = {}

    if args.dataset_name == "mnist":
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

    


