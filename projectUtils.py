from enum import Enum


class DatasetType(Enum):
    TRAINING = "train"
    VALIDATION = "val"
    TEST = "test"

class UseDataset(Enum):
    BASE = "base"
    ADVERSARIAL = "adversarial"
    COMBINED = "combined"

class ModelMode(Enum):
    TRAINING = "training"
    TEST = "test"


MODELS_DIR = "models/resnet18_base/"
ADVERSARIAL_DATASET_PATH = "adversarial"
IMAGE_FILENAME_FORMAT = "adv_{index}.png"
