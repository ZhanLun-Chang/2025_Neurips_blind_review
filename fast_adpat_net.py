import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, shufflenet_v2_x1_0
import timm
from torch import Tensor
import math

class net(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        if dataset_name == "mnist" or dataset_name == "fmnist":
            self.in_channel = 28 * 28
        self.out_channel = 10

        # Define the main network (pre-output layers)
        self.feature_extractor = nn.Linear(self.in_channel, self.out_channel)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        features = self.feature_extractor(x)  # Features before the output
        output = nn.functional.log_softmax(features, dim=1)
        return output
    
    def forward_features(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        features = self.feature_extractor(x)  
        return features

class FastText(nn.Module):
    """
    A FastText-like model for text classification using averaged word embeddings.
    Includes a forward_features method returning the averaged embeddings.
    """
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super(FastText, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pad_idx = pad_idx # Store pad_idx to exclude from average

        # In the original FastText spirit, there isn't a separate dropout layer
        # before the final linear layer on the averaged features.
        # If you needed dropout, you'd typically apply it to the averaged_embeddings
        # within the main forward method if desired, but it's less common than in CNNs/RNNs.
        # We will stick to the core FastText idea for the forward_features.

        # Linear layer for classification
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward_features(self, text):
        """
        Computes the averaged embeddings for the input text.
        Equivalent to the feature extraction part before the final classifier.

        Args:
            text (torch.Tensor): Input tensor of shape (batch_size, seq_len)
                                representing token indices.

        Returns:
            torch.Tensor: Averaged embeddings of shape (batch_size, embed_dim).
        """
        # text shape: (batch_size, seq_len)
        embedded = self.embedding(text) # shape: (batch_size, seq_len, embed_dim)

        # Create a mask to exclude padding tokens from the average
        mask = (text != self.pad_idx).float().unsqueeze(-1) # shape: (batch_size, seq_len, 1)

        # Apply the mask to the embeddings (sets embeddings of padding tokens to 0)
        masked_embeddings = embedded * mask # shape: (batch_size, seq_len, embed_dim)

        # Sum the masked embeddings across the sequence length dimension
        summed_embeddings = torch.sum(masked_embeddings, dim=1) # shape: (batch_size, embed_dim)

        # Count the number of non-padding tokens per sequence for averaging
        sequence_lengths = mask.sum(dim=1) # shape: (batch_size, 1)

        # Expand sequence_lengths to match the dimensions of summed_embeddings for division
        sequence_lengths = sequence_lengths.expand_as(summed_embeddings) # shape: (batch_size, embed_dim)

        # Add a small epsilon to prevent division by zero
        epsilon = 1e-6
        averaged_embeddings = summed_embeddings / (sequence_lengths + epsilon) # shape: (batch_size, embed_dim)

        # These averaged_embeddings are the features before the final linear layer
        return averaged_embeddings

    def forward(self, text):
        """
        Forward pass of the FastText model, using forward_features.

        Args:
            text (torch.Tensor): Input tensor of shape (batch_size, seq_len)
                                representing token indices.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
                          representing class scores.
        """
        # Get the features by calling forward_features
        features = self.forward_features(text) # shape: (batch_size, embed_dim)

        # Apply the final linear layer to the features
        # No dropout applied to features in standard basic FastText
        return self.fc(features)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv1 = nn.Conv2d(1, 100, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, embed_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, embed_dim))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward_features(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # (B, 1, L, E)

        x1 = F.relu(self.conv1(x)).squeeze(3)
        x2 = F.relu(self.conv2(x)).squeeze(3)
        x3 = F.relu(self.conv3(x)).squeeze(3)

        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        features = torch.cat((x1, x2, x3), 1)  # shape: (B, 300)
        return features  # without dropout or final linear layer

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.fc(x)
    

class SVHN_Cons(nn.Module):
    def __init__(self, hidden_dims, projection_dim=256, output_dim=10):
        super(SVHN_Cons, self).__init__()
        
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)  # BatchNorm for better convergence
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Pooling and activation
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # Projection layers for contrastive learning
        self.l1 = nn.Linear(hidden_dims[1], hidden_dims[1])  # Hidden layer refinement
        self.l2 = nn.Linear(hidden_dims[1], projection_dim)  # Dimensionality reduction
        self.l3 = nn.Linear(projection_dim, output_dim)  # Final classification layer

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Normalization for contrastive embeddings
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Dynamically flatten
        
        # Fully connected layers with Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Projection layers
        x = self.relu(self.l1(x))  # Hidden layer refinement
        x = self.l2(x)  # Dimensionality reduction
        x = self.norm(x)  # Normalization for contrastive embeddings
        
        # Classification layer
        x = self.l3(x)  # Final output logits for classification
        
        return x

    def forward_features(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool

        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Projection Layers
        x = self.relu(self.l1(x))
        x = self.l2(x)
        x = self.norm(x)  # Normalized embeddings for contrastive tasks

        return x  # Return embeddings before the final classification layer


class CNN_SVHN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Res18Con(nn.Module):
    def __init__(self, out_dim, n_classes):

        super(Res18Con, self).__init__()
        basemodel = resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        self.l3 = nn.Linear(out_dim, n_classes)
    
    def forward_features(self, x):
        h = self.features(x)
        h = h.squeeze()
        #print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return x

    def forward(self, x):
        h = self.features(x)
        #print("h after:", h)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        y = self.l3(x)
        return y

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-separable convolution block:
      1) Depthwise conv → BN → ReLU6
      2) Pointwise conv → BN → ReLU6
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SVHNFastContrastiveNet(nn.Module):
    """
    Lightweight SVHN classifier with a contrastive‐learning projection head.
    """

    def __init__(
        self,
        projection_dim: int = 256,
        num_classes: int = 10,
        dropout_prob: float = 0.5
    ) -> None:
        super().__init__()

        # === Feature extractor (MobileNetV2‐inspired) ===
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2, padding=1),  # downsample
            DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=2, padding=1),  # downsample
            DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # === Projection head for contrastive learning ===
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # === Classification head ===
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Extracts and projects features for contrastive learning.
        Returns a normalized embedding of size `projection_dim`.
        """
        x = self.stem(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)               # flatten
        x = self.proj(x)                        # projection + norm
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Full forward pass: feature extraction → projection → classification logits.
        """
        embedding = self.forward_features(x)
        logits = self.classifier(embedding)
        return logits

class MobileNetV3SmallFeatureExtractor(nn.Module):
    """
    MobileNetV3-Small backbone with separate heads for:
      1) contrastive feature projection
      2) standard classification.
    Dynamically infers the backbone’s feature dimension from a 32×32 dummy input.

    Methods
    -------
    forward_features(x)
        Returns projection_dim features for contrastive losses.
    forward(x)
        Returns logits for standard classification.
    """

    def __init__(
        self,
        num_classes: int = 10,
        projection_dim: int = 256,
        pretrained: bool = True,
        backbone_name: str = "mobilenetv3_small_100"
    ) -> None:
        super().__init__()

        # --- 1) Load backbone without head ---
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0          # strip off any classifier
            )
            state = "pre-trained" if pretrained else "from scratch"
            print(f"Loaded {state} {backbone_name} backbone (no head).")
        except Exception as e:
            raise RuntimeError(f"Failed to load {backbone_name}: {e}")

        # --- 2) Infer feature_dim using a 32×32 dummy (SVHN) ---
        device = next(self.backbone.parameters()).device

        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32, device=device)
            feats = self.backbone(dummy)
        # restore original mode
        if was_training:
            self.backbone.train()

        self.feature_dim = feats.shape[1]
        print(f"Inferred backbone feature_dim = {self.feature_dim}")

        # --- 3) Projection head for contrastive learning ---
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim)
        )
        print(f"Projection head: {self.feature_dim} → {projection_dim}")

        # --- 4) Classification head for SVHN (10 classes) ---
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        print(f"Classification head: {self.feature_dim} → {num_classes}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute projection features for contrastive loss.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 3, 32, 32)
        Returns
        -------
        torch.Tensor of shape (B, projection_dim)
        """
        h = self.backbone(x)
        return self.projection_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 3, 32, 32)
        Returns
        -------
        torch.Tensor of shape (B, num_classes)
        """
        h = self.backbone(x)
        return self.classifier(h)

class MobileNetV2FeatureExtractor(nn.Module):
    """
    MobileNetV2 backbone with separate heads for contrastive feature
    projection and standard classification. Dynamically infers the true
    backbone feature dimension for flexible head construction.

    Methods
    -------
    forward_features(x)
        Returns projection_dim features for contrastive losses.
    forward(x)
        Returns logits for standard classification.
    """

    def __init__(
        self,
        num_classes: int = 10,
        projection_dim: int = 256,
        pretrained: bool = True,
        backbone_name: str = "mobilenetv2_100"
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes for the classification head.
        projection_dim : int
            Dimension of the projection head output for contrastive learning.
        pretrained : bool
            If True, loads ImageNet-pretrained weights into the backbone.
        backbone_name : str
            Model identifier for timm.create_model (e.g., "mobilenetv2_100").
        """
        super().__init__()

        # Load backbone without any classifier head
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0
            )
            state = "pre-trained" if pretrained else "scratch"
            print(f"Loaded {state} {backbone_name} backbone (no classifier).")
        except Exception as e:
            raise RuntimeError(f"Failed to load {backbone_name}: {e}")

        # Infer true feature dimension via dummy forward pass
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            features = self.backbone(dummy)
        self.feature_dim = features.shape[1]
        print(f"Inferred backbone feature_dim = {self.feature_dim}")

        # Contrastive projection head: feature_dim -> feature_dim -> projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim)
        )
        print(f"Projection head: {self.feature_dim} -> {projection_dim}")

        # Classification head: feature_dim -> num_classes
        self.direct_classifier = nn.Linear(self.feature_dim, num_classes)
        print(f"Classification head: {self.feature_dim} -> {num_classes}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute projection features for contrastive learning.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W).

        Returns
        -------
        torch.Tensor
            Projected features of shape (batch_size, projection_dim).
        """
        h = self.backbone(x)
        return self.projection_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits for standard classification.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, num_classes).
        """
        h = self.backbone(x)
        return self.direct_classifier(h)


class MobileNetV3LargeFeatureExtractor(nn.Module):
    """
    MobileNetV3-Large backbone with separate heads for contrastive feature
    projection and standard classification.

    - forward_features(x) returns a projection_dim feature vector suitable
      for contrastive losses (e.g., SimCLR, MOON).
    - forward(x) returns class logits for num_classes.

    Attributes:
        feature_dim (int): Dimension of the features output by the backbone.
        projection_head (nn.Sequential): MLP mapping features to projection_dim.
        direct_classifier (nn.Linear): Linear layer mapping features to num_classes.
    """

    def __init__(
        self,
        num_classes: int = 100,
        projection_dim: int = 256,
        pretrained: bool = True,
        backbone_name: str = "mobilenetv3_large_100"
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Number of classes for the classification head.
        projection_dim : int
            Output dimension of the projection head for contrastive learning.
        pretrained : bool
            If True, loads ImageNet-pretrained weights into the backbone.
        backbone_name : str
            Identifier for timm.create_model.
        """
        super().__init__()

        # Load backbone without any classifier head
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0
            )
            msg = (f"Loaded pre-trained {backbone_name}" if pretrained
                   else f"Initialized {backbone_name} from scratch")
            print(f"{msg} without classifier head.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {backbone_name} from timm: {e}"
            )

        # Infer feature dimension via a dummy forward pass
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            features = self.backbone(dummy)
        self.feature_dim = features.shape[1]
        print(f"Inferred backbone feature_dim = {self.feature_dim}.")

        # Projection head: feature_dim -> feature_dim -> projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim)
        )
        print(
            f"Projection head: {self.feature_dim} -> {self.feature_dim} -> {projection_dim}"
        )

        # Standard classification head: feature_dim -> num_classes
        self.direct_classifier = nn.Linear(self.feature_dim, num_classes)
        print(
            f"Classification head: {self.feature_dim} -> {num_classes}"
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute projected features for contrastive learning.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (batch_size, projection_dim).
        """
        h = self.backbone(x)
        return self.projection_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits for standard classification.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, num_classes).
        """
        h = self.backbone(x)
        return self.direct_classifier(h)
    

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet model (18 or 34) loaded from timm, with a projection head and feature extraction method.
    Suitable for algorithms like MOON or SimCLR that require intermediate features
    for contrastive learning, while also providing a standard classification output.
    """
    def __init__(self, resnet_version=18, num_classes=10, projection_dim=256, pretrained=True):
        """
        Initializes the ResNetFeatureExtractor model using timm.

        Args:
            resnet_version (int): The version of ResNet to use (18 or 34).
            num_classes (int): The number of output classes for the final classification layer.
                               (e.g., 10 for CIFAR10, 100 for CIFAR100, 200 for Tiny ImageNet).
            projection_dim (int): The output dimension of the projection head.
                                  This is the size of the feature vector returned by forward_features.
            pretrained (bool): If True, loads pre-trained ImageNet weights for the backbone.
                               The projection head and final classification layer are always re-initialized.
        """
        super(ResNetFeatureExtractor, self).__init__()

        # Validate the resnet_version argument
        if resnet_version not in [18, 34]:
            raise ValueError("resnet_version must be either 18 or 34")

        # Load the base ResNet model backbone using timm
        # Construct the model name based on the chosen version
        model_name = f'resnet{resnet_version}' # Specify the model name in timm
        try:
            self.resnet_backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0 # Set num_classes=0 to remove the default classifier layer
            )
            if pretrained:
                print(f"Loaded pre-trained {model_name} backbone from timm.")
            else:
                print(f"Initialized {model_name} backbone from timm from scratch.")
        except Exception as e:
            print(f"Error loading model {model_name} from timm: {e}")
            print("Please check if the model name is correct for your timm version.")
            raise # Re-raise the exception after printing the message


        # Get the number of input features to the original final FC layer
        # In timm models with num_classes=0, the attribute 'num_features'
        # typically holds the dimension of the global average pooled features.
        num_ftrs = self.resnet_backbone.num_features

        # Define the projection head (often a MLP)
        # This maps the features from the backbone to a lower-dimensional space
        # suitable for contrastive learning.
        # A common structure is Linear -> ReLU -> Linear
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs), # First layer
            nn.ReLU(inplace=True),        # ReLU activation
            nn.Linear(num_ftrs, projection_dim) # Output layer with projection_dim features
            # Optional: Add a normalization layer here if required by the specific
            # contrastive loss function (e.g., nn.BatchNorm1d(projection_dim))
        )
        print(f"Projection head defined with output dimension {projection_dim}.")

        # Define the final classification layer
        # This maps the projected features to the number of classes for the
        # standard supervised classification task.
        self.classifier_layer = nn.Linear(projection_dim, num_classes)
        print(f"Final classification layer adapted for {num_classes} classes.")


    def forward_features(self, x):
        """
        Extracts features from the model after the projection head.
        This output is typically used for contrastive learning objectives.

        Args:
            x (torch.Tensor): Input tensor (image batch).

        Returns:
            torch.Tensor: Feature tensor after the projection head.
        """
        # Pass through the timm backbone.
        # When num_classes=0, timm's forward usually returns the features
        # before the classifier (after global pooling).
        h = self.resnet_backbone(x)

        # Pass the pooled features through the projection head
        features = self.projection_head(h)

        # If a normalization layer was added in the projection_head, it's applied here.
        # If your MOON/contrastive loss requires L2 normalization, you might apply it here:
        # features = F.normalize(features, dim=1)

        return features # Return features after the projection head

    def forward(self, x):
        """
        Standard forward pass through the entire model for classification.

        Args:
            x (torch.Tensor): Input tensor (image batch).

        Returns:
            torch.Tensor: Output logits for classification.
        """
        # Get the features after the projection head
        projected_features = self.forward_features(x)

        # Pass the projected features through the final classification layer
        logits = self.classifier_layer(projected_features)

        return logits
    

class SVHNEfficient(nn.Module):
    """
    Efficient CNN backbone for SVHN classification.

    Architecture:
      - Three convolutional blocks (Conv→BN→ReLU×2 → MaxPool)
      - Two-layer MLP classifier with dropout
    """
    def __init__(self, output_dim: int = 10, dropout: float = 0.4):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 3×32×32 → 32×16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32×16×16 → 64×8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64×8×8 → 128×4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Compute flattened feature size (128 × 4 × 4)
        fc_in_dim = 128 * 4 * 4

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor and classifier.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).
        Returns:
            Logits tensor of shape (B, output_dim).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features before classification.

        Useful for contrastive heads or feature analysis.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.classifier[0](x), inplace=True)  # features after first FC
    

class SVHNEfficientOptimized(nn.Module):
    """
    Optimized Efficient CNN backbone for SVHN classification.

    Key Optimizations:
      - Added AdaptiveAvgPool2d before the classifier to significantly
        reduce the number of features, making the FC layers lighter.

    Architecture:
      - Three convolutional blocks:
          - Block 1 & 2: (Conv→BN→ReLU×2 → MaxPool)
          - Block 3: (Conv→BN→ReLU×1 → MaxPool) - Note: Original code has 1 Conv in Block 3
      - Adaptive Average Pooling layer
      - Two-layer MLP classifier with dropout
    """
    def __init__(self, output_dim: int = 10, dropout: float = 0.4):
        super().__init__()

        # Feature extraction
        self.features_extractor = nn.Sequential(
            # Block 1: 3×32×32 → 32×16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32×16×16 → 64×8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64×8×8 → 128×4×4
            # Note: Original code structure had one Conv layer here, which is retained.
            # Docstring in original implied two, but one is faster.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling layer to reduce feature map size before classifier
        # This significantly reduces the number of parameters in the first Linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2)) # Output: 128 x 2 x 2

        # Compute flattened feature size after adaptive pooling (128 channels * 2 * 2)
        fc_in_dim = 128 * 2 * 2 # This is now 512, down from 2048

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_dim, 256), # Input dimension is now much smaller
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor, adaptive pooling, and classifier.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).
        Returns:
            Logits tensor of shape (B, output_dim).
        """
        x = self.features_extractor(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.classifier(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features after the first FC layer of the classifier.

        Useful for contrastive heads or feature analysis.
        The input to the classifier is now post-adaptive pooling.
        """
        x = self.features_extractor(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        # Apply the first linear layer and its activation from the classifier
        # self.classifier[0] is the first Linear layer
        # self.classifier[1] is the ReLU activation
        # Note: If you only want features before ReLU, just use self.classifier[0](x)
        features_after_first_fc_block = self.classifier[1](self.classifier[0](x))
        return features_after_first_fc_block

class EfficientNetFeatureExtractor(nn.Module):
    """
    EfficientNet-B0 model loaded from timm, with a projection head and feature extraction method.
    Suitable for contrastive learning (e.g., MOON, SimCLR) and also provides classification output.
    """
    def __init__(self, num_classes=10, projection_dim=256, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes for classification.
            projection_dim (int): Dimension of the projection head output.
            pretrained (bool): If True, load ImageNet-pretrained weights for the backbone.
        """
        super().__init__()

        # Load the EfficientNet-B0 backbone without its classifier
        model_name = 'efficientnet_b0'
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # remove default classification head
            )
            if pretrained:
                print(f"Loaded pre-trained {model_name} backbone from timm.")
            else:
                print(f"Initialized {model_name} backbone from timm from scratch.")
        except Exception as e:
            print(f"Error loading model {model_name} from timm: {e}")
            raise

        # Number of features produced by the backbone's global pooling layer
        num_ftrs = self.backbone.num_features

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, projection_dim)
            # Optionally: add normalization (e.g., nn.BatchNorm1d(projection_dim))
        )
        print(f"Projection head defined with input {num_ftrs} and output {projection_dim}.")

        # Final classifier maps projection features to classes
        self.classifier = nn.Linear(projection_dim, num_classes)
        print(f"Classifier layer adapted for {num_classes} classes.")

    def forward_features(self, x):
        """
        Extract features via backbone and projection head.
        Returns projection_dim features for contrastive loss.
        """
        # Backbone returns pooled features when num_classes=0
        h = self.backbone(x)
        features = self.projection_head(h)
        # If needed: L2 normalize features for contrastive objectives
        # features = F.normalize(features, dim=1)
        return features

    def forward(self, x):
        """
        Standard forward pass for classification.
        """
        proj = self.forward_features(x)
        logits = self.classifier(proj)
        return logits

class EfficientNetLiteFeatureExtractor(nn.Module):
    """
    EfficientNet-Lite model loaded from timm, with a projection head and feature extraction method.
    Suitable for contrastive learning (e.g., MOON, SimCLR) and also provides classification output.
    """
    def __init__(self, num_classes=10, projection_dim=256, pretrained=True, lite_variant='0'):
        """
        Args:
            num_classes (int): Number of output classes for classification.
            projection_dim (int): Dimension of the projection head output.
            pretrained (bool): If True, load ImageNet-pretrained weights for the backbone.
            lite_variant (str): The specific EfficientNet-Lite variant to use (e.g., '0', '1', '2', '3', '4').
                                 See timm documentation for available lite models.
        """
        super().__init__()

        # Construct the EfficientNet-Lite model name
        # Common variants are 'efficientnet_lite0', 'efficientnet_lite1', etc.
        model_name = f'efficientnet_lite{lite_variant}'
        print(f"Attempting to load {model_name}...")

        try:
            # Load the EfficientNet-Lite backbone without its classifier
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # remove default classification head
            )
            if pretrained:
                print(f"Successfully loaded pre-trained {model_name} backbone from timm.")
            else:
                print(f"Successfully initialized {model_name} backbone from timm from scratch.")
        except Exception as e:
            print(f"Error loading model {model_name} from timm: {e}")
            print("Please ensure the model name is correct and timm is installed with its dependencies.")
            print("Available EfficientNet-Lite models in timm might include names like:")
            print(" 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', etc., or just 'efficientnet_lite0'")
            print("You can list available models with: `timm.list_models('*efficientnet_lite*')`")
            raise

        # Number of features produced by the backbone's global pooling layer
        # This attribute name is consistent across timm models
        num_ftrs = self.backbone.num_features
        print(f"Backbone '{model_name}' has {num_ftrs} output features.")

        # Projection head for contrastive learning
        # This part remains the same as it operates on the features from the backbone
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, projection_dim)
            # Optionally: add normalization (e.g., nn.BatchNorm1d(projection_dim))
        )
        print(f"Projection head defined with input {num_ftrs} and output {projection_dim}.")

        # Final classifier maps projection features to classes
        # This also remains the same, taking the projection head's output
        self.classifier = nn.Linear(projection_dim, num_classes)
        print(f"Classifier layer adapted for {num_classes} classes.")

    def forward_features(self, x):
        """
        Extract features via backbone and projection head.
        Returns projection_dim features for contrastive loss.
        """
        # Backbone returns pooled features when num_classes=0
        h = self.backbone(x) # (batch_size, num_ftrs)
        features = self.projection_head(h) # (batch_size, projection_dim)

        # If needed for specific contrastive objectives (e.g., SimCLR, MoCo):
        # L2 normalize features. For MOON, this might not be standard.
        # features = F.normalize(features, dim=1)
        return features

    def forward(self, x):
        """
        Standard forward pass for classification.
        Passes data through backbone, projection head, and finally the classifier.
        """
        # Get features from the projection head
        proj_features = self.forward_features(x) # (batch_size, projection_dim)

        # Pass projected features through the classifier
        logits = self.classifier(proj_features) # (batch_size, num_classes)
        return logits

class ShuffleNetFeatureExtractor(nn.Module):
    """
    ShuffleNetV2 (1.0×) loaded from torchvision, with a projection head and feature extraction method.
    Suitable for contrastive algorithms (e.g., MOON, SimCLR) and supervised classification.

    Attributes:
        backbone: ShuffleNetV2 trunk without its classifier.
        projection_head: MLP mapping features to projection_dim for contrastive losses.
        classifier_layer: Final linear layer mapping projected features to num_classes.
    """
    def __init__(
        self,
        num_classes: int = 10,
        projection_dim: int = 256,
        pretrained: bool = True
    ):
        super().__init__()

        # Load base ShuffleNetV2 backbone
        try:
            base_model = shufflenet_v2_x1_0(weights='DEFAULT' if pretrained else None)
            if pretrained:
                print("Loaded pre-trained ShuffleNetV2_x1_0 backbone from torchvision.")
            else:
                print("Initialized ShuffleNetV2_x1_0 backbone from scratch.")
        except Exception as e:
            print(f"Error loading ShuffleNetV2_x1_0 from torchvision: {e}")
            raise

        # Extract feature dimension from the original classifier
        num_ftrs = base_model.fc.in_features

        # Remove original classifier
        base_model.fc = nn.Identity()
        self.backbone = base_model

        # Projection head: [num_ftrs -> num_ftrs -> projection_dim]
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, projection_dim)
        )
        print(f"Projection head defined with output dimension {projection_dim}.")

        # Final classification layer: [projection_dim -> num_classes]
        self.classifier_layer = nn.Linear(projection_dim, num_classes)
        print(f"Final classification layer adapted for {num_classes} classes.")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts projected features for contrastive learning.

        Args:
            x: Input tensor (B, 3, H, W).
        Returns:
            Tensor of shape (B, projection_dim).
        """
        # Backbone outputs flattened pooled features
        h = self.backbone(x)
        # Projection head
        features = self.projection_head(h)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for classification.

        Args:
            x: Input tensor (B, 3, H, W).
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        projected = self.forward_features(x)
        logits = self.classifier_layer(projected)
        return logits

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts and L2-normalizes features (useful for some contrastive losses).
        """
        feats = self.forward_features(x)
        return F.normalize(feats, dim=1)


class GhostModule(nn.Module):
    """
    Ghost module:
      - Primary 1×1 conv generates intrinsic features.
      - Cheap depthwise conv generates additional "ghost" features.
    """
    def __init__(self, in_channels: int, out_channels: int, ratio: int = 2):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        ghost_channels = init_channels * (ratio - 1)
        # primary conv
        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        # cheap operation (depthwise conv)
        self.cheap = nn.Sequential(
            nn.Conv2d(init_channels, ghost_channels, kernel_size=3, stride=1, padding=1,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(ghost_channels),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary_feat = self.primary(x)
        ghost_feat = self.cheap(primary_feat)
        out = torch.cat([primary_feat, ghost_feat], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostCNN(nn.Module):
    """
    GhostCNN backbone for SVHN classification.

    Architecture:
      - Three GhostModule blocks, each followed by MaxPool2d
      - Two-layer MLP classifier with dropout
    """
    def __init__(self, output_dim: int = 10, dropout: float = 0.2):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 3×32×32 → GhostModule → MaxPool → 32×16×16
            GhostModule(3, 32),
            nn.MaxPool2d(2),

            # Block 2: 32×16×16 → GhostModule → MaxPool → 64×8×8
            GhostModule(32, 64),
            nn.MaxPool2d(2),

            # Block 3: 64×8×8 → GhostModule → MaxPool → 128×4×4
            GhostModule(64, 128),
            nn.MaxPool2d(2)
        )

        # Compute flattened feature size (128 × 4 × 4)
        fc_in_dim = 128 * 4 * 4

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor and classifier.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).
        Returns:
            Logits tensor of shape (B, output_dim).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features before classification.

        Useful for contrastive heads or feature analysis.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def ds_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """
    Depthwise separable convolution block:
      - Depthwise 3×3 conv
      - BatchNorm + ReLU
      - Pointwise 1×1 conv
      - BatchNorm + ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class DepthwiseSeparableCNN(nn.Module):
    """
    DepthwiseSeparableCNN backbone for SVHN classification.

    Architecture:
      - Four depthwise-separable blocks, each optionally downsamples via stride
      - Two-layer MLP classifier with dropout
    """
    def __init__(self, output_dim: int = 10, dropout: float = 0.2):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 3×32×32 → 32×32×32
            ds_block(3, 32, stride=1),
            # Block 2: 32×32×32 → 64×16×16
            ds_block(32, 64, stride=2),
            # Block 3: 64×16×16 → 128×8×8
            ds_block(64, 128, stride=2),
            # Block 4: 128×8×8 → 256×4×4
            ds_block(128, 256, stride=2)
        )

        # Compute flattened feature size (256 × 4 × 4)
        fc_in_dim = 256 * 4 * 4

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor and classifier.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).
        Returns:
            Logits tensor of shape (B, output_dim).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features before classification.

        Useful for contrastive heads or feature analysis.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
