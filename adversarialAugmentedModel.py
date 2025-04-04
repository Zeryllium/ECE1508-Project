import logging
import os

import torch

from main import run
from main import setup_logger
from projectUtils import ModelMode
from projectUtils import UseDataset

if __name__ == "__main__":
    model_name = "resnet18_adversarial_augmented"
    model_save_path = "./models"

    # This is unused when ModelMode.TRAINING but you need it when ModelMode.TEST
    model_filename = "2025-04-03T18-53_epoch_5.pth"

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
        "random_augment": True,
        "batch_size": 64,
        "epochs": 20,
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
        "lr": 1e-3,
        "model_name": model_name,
        "model_save_path": model_save_path,
        "model_filename": model_filename,
    }

    run(params, logger, start_time)