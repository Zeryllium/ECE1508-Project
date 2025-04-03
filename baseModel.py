import logging
import os

import torch

from main import run
from main import setup_logger
from projectUtils import ModelMode
from projectUtils import UseDataset

if __name__ == "__main__":
    model_name = "resnet18_base"
    model_save_path = "./models"

    model_filename = "2025-04-02T17-57_epoch_4.pth"

    logger, start_time = setup_logger(
        logfile_path= os.path.join(
            model_save_path,
            model_name
        ),
        log_level=logging.INFO
    )

    params ={
        "mode": ModelMode.TEST,
        "dataset": UseDataset.COMBINED,
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