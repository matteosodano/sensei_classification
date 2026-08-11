import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def create_datasets(path_to_data):
    train_dataset = ClassificationDataset(
        os.path.join(path_to_data, "train"),
        image_size=(224, 224)
    )

    val_dataset = ClassificationDataset(
        os.path.join(path_to_data, "validation"),
        class_to_idx=train_dataset.class_to_idx,
        image_size=(224, 224)
    )

    return train_dataset, val_dataset

def create_loaders(train_dataset, val_dataset):
    train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4)

    val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4)

    return train_loader, val_loader

class ClassificationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        class_to_idx=None,
        image_size=(224, 224),
        transform=None,
    ):
        """
        Args:
            root_dir: Path to train/ or validation/
            class_to_idx: Existing class mapping. If None, create it
                          from the classes found in root_dir.
            image_size: (height, width) used to resize images.
            transform: Optional additional torchvision transforms.
        """

        self.root_dir = root_dir

        # Find class folders
        class_names = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

        # If no mapping is provided, create it.
        # This is normally done for the training set.
        if class_to_idx is None:
            self.class_to_idx = {
                class_name: idx
                for idx, class_name in enumerate(class_names)
            }
        else:
            # Copy it so we don't modify the original mapping
            self.class_to_idx = dict(class_to_idx)

            # Add classes that are only present in this dataset
            next_idx = max(self.class_to_idx.values(), default=-1) + 1

            for class_name in class_names:
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = next_idx
                    next_idx += 1

        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Collect all (image_path, class_id) pairs
        self.samples = []

        image_extensions = {
            ".jpg", ".jpeg", ".png",
            ".bmp", ".webp", ".tif", ".tiff"
        }

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_id = self.class_to_idx[class_name]

            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)

                if not os.path.isfile(image_path):
                    continue

                extension = os.path.splitext(filename)[1].lower()

                if extension in image_extensions:
                    self.samples.append(
                        (image_path, class_id)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_id = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, class_id