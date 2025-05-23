import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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
    