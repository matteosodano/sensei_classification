import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model(backbone, pretrained, train_classes):
    if backbone == 'resnet':
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)

        # Replace ImageNet classifier
        num_features = model.fc.in_features
        model.fc = nn.Linear(
            num_features,
            len(train_classes)
        )
    elif backbone == "dino":
        model = DINOClassifier(
            num_classes=len(train_classes),
            model_name="dinov2_vits14",
            pretrained=pretrained,
        )
    else:
        raise ValueError(
            f"Unknown backbone: {backbone}"
        )
    return model


class DINOClassifier(nn.Module):
    """
    DINOv2 backbone with a trainable linear classification head.
    The DINOv2 backbone is frozen and only the classifier is trained.
    """
    def __init__(
        self,
        num_classes,
        model_name="dinov2_vits14",
        pretrained=True,
    ):
        super().__init__()

        # Load DINOv2
        if pretrained:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
            )
        else:
            raise NotImplementedError(
                "Non-pretrained DINOv2 is not currently supported."
            )

        # Freeze DINOv2
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

        # DINOv2 ViT-S/14 has 384-dimensional features
        feature_dim = self.backbone.embed_dim

        # Trainable classifier
        self.classifier = nn.Linear(
            feature_dim,
            num_classes,
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)

        logits = self.classifier(features)

        return logits

    def train(self, mode=True):
        """
        Keep DINOv2 in eval mode even when the whole model
        is switched to train mode!
        """
        super().train(mode)

        self.backbone.eval()

        return self