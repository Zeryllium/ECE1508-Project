import logging
import os
from copy import deepcopy
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.datasets as DS
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from adversarialDataset import AdversarialDataset
from projectUtils import DatasetType
from projectUtils import ModelMode
from projectUtils import UseDataset
from resnet18 import ResNet18


def test_model_single_sample(model, image, label, device):
    model.eval()

    image = image.to(device)

    prediction = model.forward(image.unsqueeze(0)).squeeze()

    print(f"Confidence level per class: {torch.nn.functional.softmax(prediction, dim=0)}")
    print(f"Predicted label: {torch.argmax(prediction)} | True label: {label}")

    return

if __name__ == "__main__":
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    source_image = transforms(plt.imread("source_sample.png"))
    adversarial_image = transforms(plt.imread("adversarial_sample.png"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18().to(device)
    model.load_state_dict(torch.load("models/resnet18_base/2025-04-03T16-37_epoch_4.pth"))

    print("Regular sample")
    test_model_single_sample(model, source_image, "8", device)
    print("Adversarial sample")
    test_model_single_sample(model, adversarial_image, "8", device)