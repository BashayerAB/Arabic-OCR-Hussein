import torch.nn as nn
import torchvision.models as models

class EfficientNetRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetRNNModel, self).__init__()

        # Load EfficientNet B0 as feature extractor
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Remove the last fully connected layer, we only need the feature extractor
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-2])

        # Classifier layer (matches the saved weights)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        # EfficientNet feature extractor
        x = self.feature_extractor(x)

        # Global Average Pooling to reduce spatial dimensions
        x = x.mean([2, 3])  # Average over height and width dimensions

        # Classifier
        x = self.classifier(x)
        return x
