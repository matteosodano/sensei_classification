import torch
import torch.nn as nn
import torch.optim as optim
import click

from dataset import create_datasets, create_loaders
from models import get_model
from train_loop import train
from val_loop import validate
from test_loop import gondola_evaluation
from utils import check_mappings


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option(
    "--path_to_data",
    default="./data",
    type=str,
    show_default=True,
    help="Path to folder containing 'train' and 'validation' subfolders."
)
@click.option(
    "--lr",
    default=1e-4,
    type=float,
    show_default=True,
    help="Learning rate."
)
@click.option(
    "--backbone",
    default="resnet",
    type=str,
    show_default=True,
    help="Backbone architecture."
)
@click.option(
    "--pretrained",
    default=True,
    type=bool,
    show_default=True,
    help="Whether to use pretrained weights."
)
@click.option(
    "--path",
    default="best.pth",
    type=str,
    show_default=True,
    help="Path to save the best model."
)
@click.option(
    "--threshold",
    default=0.6,
    type=float,
    show_default=True,
    help="Confidence threshold used during validation."
)
def main(path_to_data, lr, backbone, pretrained, path, threshold):
    """Train, validate, and evaluate the object classification model."""

    train_dataset, val_dataset = create_datasets(path_to_data)

    check_mappings(train_dataset, val_dataset)

    train_loader, val_loader = create_loaders(
        train_dataset,
        val_dataset
    )

    train_classes = sorted(
        train_dataset.class_to_idx.values()
    )

    model = get_model(
        backbone,
        pretrained,
        train_classes
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr
    )

    train(
        model,
        train_loader,
        optimizer,
        criterion,
        train_classes,
        DEVICE,
        path
    )

    validate(
        model,
        val_loader,
        train_classes,
        DEVICE,
        threshold
    )

    gondola_evaluation(
        model,
        train_dataset,
        DEVICE
    )


if __name__ == "__main__":
    main()