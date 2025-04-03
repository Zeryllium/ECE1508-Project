import logging
import os

import pandas as pd
import torch
import torchvision.utils as vutils
import torchvision.transforms.v2 as transforms

from main import denormalize
from main import setup
from main import setup_logger
from projectUtils import ADVERSARIAL_DATASET_PATH
from projectUtils import IMAGE_FILENAME_FORMAT
from projectUtils import MODELS_DIR


def fgsm(image, epsilon, gradient):
    modified_image = image + epsilon * gradient.sign()
    # Clamp the images to the same input normalization (between 0 and 1)
    return torch.clamp(modified_image, 0, 1)

def generate_adversarial_samples(params, logger):
    loaders, model, device = setup(params)
    loss_fn = params.get("loss_fn")

    model.to(device)
    model.load_state_dict(torch.load(params.get("model_save_path")))
    model.eval()
    logger.debug(f"Model loaded to device: {device}")

    renormalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for dataset_type, loader in loaders.items():
        os.makedirs(os.path.join(ADVERSARIAL_DATASET_PATH, dataset_type), exist_ok=True)
        logger.info(f"Dataset {dataset_type} size: {len(loader.dataset)}")
        dataset_mapping = []

        for index, (image, label) in enumerate(loader):
            image = image.to(device)
            label = label.to(device)

            image.requires_grad = True
            prediction = model.forward(image)
            pred_label = torch.argmax(torch.nn.functional.softmax(prediction, dim=1), dim=1)

            logger.debug(f"Model output: {pred_label} | Label: {label}")

            if pred_label != label:
                logger.debug("Incorrect prediction, skipping FGSM for this sample")
                continue
            else:
                loss = loss_fn(prediction, label)
                model.zero_grad()
                loss.backward()

                adversarial_sample_prenorm = fgsm(denormalize(image), params.get("epsilon"), image.grad.data)
                adversarial_sample = renormalize(adversarial_sample_prenorm)
                adversarial_prediction = model.forward(adversarial_sample)
                adversarial_pred_label = torch.argmax(torch.nn.functional.softmax(adversarial_prediction, dim=1), dim=1)
                logger.debug(f"Adversarial output: {adversarial_pred_label}")

                # Show the image pre-fgsm and post-fgsm
                # Note that we only care about initially correctly predicted samples that are later incorrectly classified
                if adversarial_pred_label != label:
                    # Save the adversarial images and labels

                    image_path = os.path.join(
                        ADVERSARIAL_DATASET_PATH,
                        dataset_type,
                        IMAGE_FILENAME_FORMAT.format(index=index),
                    )
                    vutils.save_image(adversarial_sample_prenorm, image_path)
                    dataset_mapping.append(
                        {
                            "filename": IMAGE_FILENAME_FORMAT.format(index=index),
                            "label": label.item()
                        }
                    )
                    logger.debug(f"Adversarial image of true label {label.item()} saved to {image_path}")
                else:
                    logger.debug("FGSM failure -> skipping images")

        mapping_path = os.path.join(ADVERSARIAL_DATASET_PATH, dataset_type, "mapping.csv")
        logger.info(f"Saving adversarial sample-label mapping to {mapping_path}")
        pd.DataFrame(dataset_mapping).to_csv(mapping_path, index=False)


if __name__ == '__main__':
    logger, start_time = setup_logger(ADVERSARIAL_DATASET_PATH, logging.INFO)
    model_name = "2025-04-02T17-57_epoch_4.pth"

    params ={
        "batch_size": 1, # Ensure each datapoint has gradients independent of each other
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam,
        "lr": 1e-3,
        "model_save_path": os.path.join(
            MODELS_DIR,
            model_name
        ),
        "epsilon": 0.01,
    }

    logger.info("Generating adversarial samples.")

    generate_adversarial_samples(params, logger)

