import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.datasets as DS
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def imshow(img):
    # remove the impact of normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print(img.shape)

    img = transforms.Normalize(mean=-(mean/std), std=1/std)(img)

    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def setup():
    torch.manual_seed(1504)
    np.random.seed(1504)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_val_dataset = DS.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = DS.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    # resnet18 has a final FC layer with 1000 output features, corresponding to the 1000 image classes in ImageNet
    # replace this with 10 output features
    #model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

    if torch.backends.cuda.is_built():
        # if we have cuda
        # usually on Windows machines with GPU
        device = "cuda"
    elif torch.backends.mps.is_built():
        # if we have MPS
        # usually on MAC
        device = "mps"
    else:
        # if not we should use our CPU
        device = "cpu"

    return train_loader, val_loader, test_loader, model, device

def run(params):
    print(f"Running with parameters: {params}")
    train_loader, val_loader, test_loader, model, device = setup()
    print(f"Device: {device}")

    model.to(device)
    optimizer = params.get("optimizer")(model.parameters(), lr=params.get("lr"))
    loss_fn = params.get("loss_fn")

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []

    epoch_best = ()

    for epoch in range(params.get("epochs")):
        model.train()
        train_loss = 0
        train_accuracy = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            batch_predictions = model.forward(batch_images)
            batch_loss = loss_fn(batch_predictions, batch_labels)
            batch_accuracy = torch.count_nonzero(torch.eq(
                torch.argmax(torch.nn.functional.softmax(batch_predictions, dim=1), dim=1),
                batch_labels
            ))

            train_loss += batch_loss.item()
            train_accuracy += batch_accuracy.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        epoch_train_loss.append(train_loss/len(train_loader.dataset))
        epoch_train_acc.append(train_accuracy/len(train_loader.dataset))

        model.eval()
        val_loss = 0
        val_accuracy = 0

        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            batch_predictions = model.forward(batch_images)
            batch_loss = loss_fn(batch_predictions, batch_labels)
            batch_accuracy = torch.count_nonzero(torch.eq(
                torch.argmax(torch.nn.functional.softmax(batch_predictions, dim=1), dim=1),
                batch_labels
            ))

            val_loss += batch_loss.item()
            val_accuracy += batch_accuracy.item()
        epoch_val_loss.append(val_loss / len(val_loader.dataset))
        epoch_val_acc.append(val_accuracy / len(val_loader.dataset))

        print(f"Epoch: {epoch} | Train Loss: {epoch_train_loss[-1]:.4f} | Train Acc: {epoch_train_acc[-1]:.4f} || Val Loss: {epoch_val_loss[-1]:.4f} | Val Acc: {epoch_val_acc[-1]:.4f}")

        if np.argmin(epoch_val_loss) == epoch:
            epoch_best = (deepcopy(model.state_dict()), epoch)


    # plot the training and test losses
    plt.plot([i for i in range(params.get("epochs"))], epoch_train_loss, label='Training')
    plt.plot([i for i in range(params.get("epochs"))], epoch_val_loss, label='Validation')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # plot the test accuracy
    plt.plot([i+1 for i in range(params.get("epochs"))], epoch_train_acc, label='Training')
    plt.plot([i+1 for i in range(params.get("epochs"))], epoch_val_acc, label='Validation')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    print(f"Saving epoch best: {epoch_best[1]}")
    save_path = params.get("model_save_path").format(model_name=params.get("model_name"))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    torch.save(epoch_best[0], os.path.join(save_path, f"epoch_{epoch_best[1]}.pth"))

if __name__ == "__main__":
    # TODO: set up command-line argparse
    params ={
        "epochs": 20,
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam,
        "lr": 1e-3,
        "model_name": "resnet18_base",
        "model_save_path": "./models/{model_name}/",
    }

    run(params)
