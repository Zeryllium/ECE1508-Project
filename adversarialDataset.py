import os

import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import decode_image

from projectUtils import ADVERSARIAL_DATASET_PATH
from projectUtils import DatasetType


class AdversarialDataset(Dataset):
    """
    Adversarial Dataset generated using FGSM from a ResNet18 model trained off the CIFAR-10 dataset
    """

    def __init__(self, dataset_type:DatasetType, transform=None):
        self.dataset_path = os.path.join(ADVERSARIAL_DATASET_PATH, dataset_type.value)
        self.mappings = pd.read_csv(os.path.join(self.dataset_path, "mapping.csv"))

        # Ignore the transforms, process it manually
        # self.transform = transform
        self.transform = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.mappings)

    def __getitem__(self, idx):
        # Filename | Label
        image_mapping = self.mappings.iloc[idx]

        image_filename = os.path.join(
            self.dataset_path,
            image_mapping["filename"]
        )
        # For some reason, Pytorch loads this image as a uint8 but expects float32 when it tries using transforms
        image = decode_image(image_filename)/255
        label = image_mapping["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
