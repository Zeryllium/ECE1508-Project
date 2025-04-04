import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as transforms

from resnet18 import ResNet18
from main import setup
from main import test_model
from projectUtils import UseDataset

def test_model_d_sample(params):
    setup(params)
    return

if __name__ == "__main__":
    params = {
        "batch_size": 64,
        "dataset": UseDataset.ADVERSARIAL
    }
    loaders, model, device = setup(params)
    model.to(device)
    test_loader = loaders["test"]

    # Baseline
    model.load_state_dict(torch.load("models/resnet18_base/2025-04-03T16-37_epoch_4.pth"))
    print(test_model(model, test_loader, device))

    # Adversarial
    model.load_state_dict(torch.load("models/resnet18_adversarial/2025-04-03T17-43_epoch_3.pth"))
    print(test_model(model, test_loader, device))


    # Baseline + Augmented
    model.load_state_dict(torch.load("models/resnet18_base_augmented/2025-04-03T18-28_epoch_3.pth"))
    print(test_model(model, test_loader, device))

    # Adversarial + Augmented
    model.load_state_dict(torch.load("models/resnet18_adversarial_augmented/2025-04-03T18-53_epoch_5.pth"))
    print(test_model(model, test_loader, device))
