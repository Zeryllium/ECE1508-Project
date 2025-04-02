import os
import logging
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torchvision
import torchvision.datasets as DS
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transformed_image = transforms.Normalize(mean=-(mean/std), std=1/std)(image)
    return transformed_image


def imshow(image):
    # remove the impact of normalization
    denormalized_image = denormalize(image).squeeze()

    plt.imshow(np.transpose(denormalized_image, (1, 2, 0)))
    plt.show()


def setup(params):
    torch.manual_seed(1504)
    np.random.seed(1504)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    batch_size = params.get("batch_size")

    train_val_dataset = DS.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = DS.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

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

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    return loaders, model, device


def setup_logger(logfile_path, log_level):
    # Filepaths cannot have colons in them
    start_time = datetime.now().isoformat(timespec="minutes").replace(':', '-')

    os.makedirs(logfile_path, exist_ok=True)
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(
        os.path.join(
            logfile_path,
            f"{start_time}.log"
        )
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(log_level)

    return root_logger, start_time


def test_model(model, test_loader, device):
    model.eval()

    test_accuracy = 0
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        batch_predictions = model.forward(batch_images)

        batch_accuracy = torch.count_nonzero(torch.eq(
            torch.argmax(torch.nn.functional.softmax(batch_predictions, dim=1), dim=1),
            batch_labels
        ))

        test_accuracy += batch_accuracy.item()

    test_accuracy = test_accuracy / len(test_loader.dataset)

    return test_accuracy


def graph_model_performance(filename, train_loss, train_acc, val_loss, val_acc, test_loss=None, test_acc=None):
    figure_title = f"Fine-Tuned Model Performance Metrics"
    gs = gridspec.GridSpec(8, 1)
    figure = plt.figure(figsize=(10, 10))
    figure.suptitle(figure_title)

    ax1 = figure.add_subplot(gs[0:3, 0])
    ax1.plot(range(len(train_loss)), train_loss, label='Training Loss')
    ax1.plot(range(len(val_loss)), val_loss, label='Validation Loss')
    ax1.set(xlabel="Epoch", ylabel="Loss")

    ax2 = figure.add_subplot(gs[3:6, 0])
    ax2.plot(range(len(train_acc)), train_acc, label='Training Accuracy')
    ax2.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
    ax2.set(xlabel="Epoch", ylabel="Accuracy", ylim=[-0.1, 1.1])

    ax3 = figure.add_subplot(gs[6:8, 0])
    ax3.axis('off')

    best_epoch_train_loss = torch.argmin(torch.FloatTensor(train_loss)).item()
    best_epoch_val_loss = torch.argmin(torch.FloatTensor(val_loss)).item()

    best_epoch_metrics = [
        [best_epoch_train_loss, train_loss[best_epoch_train_loss], train_acc[best_epoch_train_loss]],
        [best_epoch_val_loss, val_loss[best_epoch_val_loss], val_acc[best_epoch_val_loss]]
    ]
    row_labels = ["Training", "Validation"]

    if test_loss is not None:
        ax1.scatter(best_epoch_val_loss, test_loss, c='#25d115', marker='x', label='Test Loss')
        ax2.scatter(best_epoch_val_loss, test_acc, c='#25d115', marker='x', label='Test Accuracy')
        best_epoch_metrics.append([best_epoch_val_loss, test_loss, test_acc])
        row_labels.append("Test")

    ax3.table(cellText=best_epoch_metrics, rowLabels=row_labels, colLabels=["Epoch", "Loss", "Accuracy"],
              loc="upper center")

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    figure.show()
    figure.savefig(filename)


def run(params, logger, start_time):
    logger.info(f"Running with parameters: {params}")
    loaders, model, device = setup(params)
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    logger.info(f"Device: {device}")

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

        logger.info(f"Epoch: {epoch} | Train Loss: {epoch_train_loss[-1]:.4f} | Train Acc: {epoch_train_acc[-1]:.4f} || Val Loss: {epoch_val_loss[-1]:.4f} | Val Acc: {epoch_val_acc[-1]:.4f}")

        if np.argmin(epoch_val_loss) == epoch:
            epoch_best = (deepcopy(model.state_dict()), epoch)

    logger.info(f"Saving epoch best: {epoch_best[1]}")
    save_path = os.path.join(
        params.get("model_save_path"),
        params.get("model_name"),
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    torch.save(
        epoch_best[0],
        os.path.join(
            save_path,
            f"{start_time}_epoch_{epoch_best[1]}.pth"
        )
    )

    test_accuracy = test_model(model, test_loader, device)
    logger.info(f"Model test accuracy: {test_accuracy}")

    graph_model_performance(
        filename=os.path.join(save_path, f"{start_time}_epoch_{epoch_best[1]}__graphs.png"),
        train_loss=epoch_train_loss,
        train_acc=epoch_train_acc,
        val_loss=epoch_val_loss,
        val_acc=epoch_val_acc,
    )

if __name__ == "__main__":
    model_name = "resnet18_base"
    model_save_path = "./models"

    logger, start_time = setup_logger(
        logfile_path= os.path.join(
            model_save_path,
            model_name
        ),
        log_level=logging.INFO
    )

    params ={
        "batch_size": 64,
        "epochs": 20,
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam,
        "lr": 1e-3,
        "model_name": model_name,
        "model_save_path": model_save_path,
    }

    run(params, logger, start_time)
