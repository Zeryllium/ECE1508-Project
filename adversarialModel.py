import os
import logging
import torch

from main import setup_logger
from main import run
from projectUtils import ModelMode
from projectUtils import UseDataset


if __name__ == "__main__":
    model_name = "resnet18_adversarial"
    model_save_path = "./models"

    # This is unused when ModelMode.TRAINING but you need it when ModelMode.TEST
    model_filename = "2025-04-02T20-52_epoch_2.pth"

    logger, start_time = setup_logger(
        logfile_path= os.path.join(
            model_save_path,
            model_name
        ),
        log_level=logging.INFO
    )

    params ={
        "mode": ModelMode.TEST,
        "dataset": UseDataset.ADVERSARIAL,
        "batch_size": 64,
        "epochs": 20,
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam,
        "lr": 1e-3,
        "model_name": model_name,
        "model_save_path": model_save_path,
        "model_filename": model_filename,
    }

    run(params, logger, start_time)