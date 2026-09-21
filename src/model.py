import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DRClassifier(nn.Module):
    """
    Transfer Learning Model based on EfficientNet-B0 pre-trained on ImageNet.
    Modified for 5-class Diabetic Retinopathy severity grading.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(DRClassifier, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Replace default classifier head with custom Dropout + Linear layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_target_layer(self):
        """
        Returns the final convolutional feature layer required by Grad-CAM for heatmaps.
        """
        return [self.backbone.features[-1]]


# Quick local sanity check when running file directly
if __name__ == "__main__":
    model = DRClassifier(num_classes=5, pretrained=True)
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    outputs = model(dummy_input)
    print(f"Model output shape: {outputs.shape}")  # Expecting torch.Size([2, 5])
    print("Model initialized successfully!")