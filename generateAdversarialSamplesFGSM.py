import os
import torch
import pandas as pd
import torchvision.utils as vutils
from torchvision.transforms import transforms

from main import denormalize
from main import imshow
from main import setup

MODEL_FILEPATH = "models/resnet18_base/epoch_6.pth"

# Create a folder to store adversarial samples, each directory will contain 10% of the original dataset
sample_dir = "adversarial_dataset"
train_dir = "adversarial_dataset/train"
valid_dir = "adversarial_dataset/validation"
test_dir = "adversarial_dataset/test"
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create a CSV file to store adversarial samples & their labels
train_adv_samples = []
valid_adv_samples = []
test_adv_samples = []

def fgsm(image, epsilon, gradient):
    modified_image = image + epsilon * gradient.sign()
    # Clamp the images to the same input normalization (between 0 and 1)
    return torch.clamp(modified_image, 0, 1)

def generate_adversarial_training_samples(params, idx):
    train_loader, val_loader, test_loader, model, device = setup(params)
    loss_fn = params.get("loss_fn")
    # print the size of the train_loader
    print(f"Train loader size: {len(train_loader.dataset)}")
    generate_limit = len(train_loader.dataset) // 10 # 10% of the dataset

    model.to(device)
    model.load_state_dict(torch.load(MODEL_FILEPATH))
    model.eval()
    print(f"Model loaded to device: {device}")

    renormalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #generate_limit = params.get("generate_limit", 0)

    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)

        image.requires_grad = True
        prediction = model.forward(image)
        pred_label = torch.argmax(torch.nn.functional.softmax(prediction, dim=1), dim=1)

        print(f"Model output: {pred_label} | Label: {label}")

        if pred_label != label:
            print("Incorrect prediction, skipping FGSM for this sample")
            continue
        else:
            loss = loss_fn(prediction, label)
            model.zero_grad()
            loss.backward()

            adversarial_sample = renormalize(
                fgsm(denormalize(image), params.get("epsilon"), image.grad.data)
            )
            adversarial_prediction = model.forward(adversarial_sample)
            adversarial_pred_label = torch.argmax(torch.nn.functional.softmax(adversarial_prediction, dim=1), dim=1)
            print(f"Adversarial output: {adversarial_pred_label}")

            # Show the image pre-fgsm and post-fgsm
            # Note that we only care about initially correctly predicted samples that are later incorrectly classified
            if adversarial_pred_label != label:
                #imshow(image.detach().cpu())
                #imshow(adversarial_sample.detach().cpu())
                generate_limit-=1

                # Save the adversarial images and labels
                image_path = os.path.join(train_dir, f"adv_sample_training_{idx}.png")
                vutils.save_image(image, image_path)
                train_adv_samples.append({
                    "image_path": image_path,
                    "original_label": label.item(),
                    "adversarial_label": adversarial_pred_label.item(),
                })
                idx += 1
            else:
                print("FGSM failure -> skipping images")

            if generate_limit==0:
                break

def generate_adversarial_validation_samples(params, idx):
    train_loader, val_loader, test_loader, model, device = setup(params)
    loss_fn = params.get("loss_fn")
    # print the size of the val_loader
    print(f"Validation loader size: {len(val_loader.dataset)}")
    generate_limit = len(val_loader.dataset) // 10 # 10% of the dataset

    model.to(device)
    model.load_state_dict(torch.load(MODEL_FILEPATH))
    model.eval()
    print(f"Model loaded to device: {device}")

    renormalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #generate_limit = params.get("generate_limit", 0)

    for image, label in val_loader:
        image = image.to(device)
        label = label.to(device)

        image.requires_grad = True
        prediction = model.forward(image)
        pred_label = torch.argmax(torch.nn.functional.softmax(prediction, dim=1), dim=1)

        print(f"Model output: {pred_label} | Label: {label}")

        if pred_label != label:
            print("Incorrect prediction, skipping FGSM for this sample")
            continue
        else:
            loss = loss_fn(prediction, label)
            model.zero_grad()
            loss.backward()

            adversarial_sample = renormalize(
                fgsm(denormalize(image), params.get("epsilon"), image.grad.data)
            )
            adversarial_prediction = model.forward(adversarial_sample)
            adversarial_pred_label = torch.argmax(torch.nn.functional.softmax(adversarial_prediction, dim=1), dim=1)
            print(f"Adversarial output: {adversarial_pred_label}")

            # Show the image pre-fgsm and post-fgsm
            # Note that we only care about initially correctly predicted samples that are later incorrectly classified
            if adversarial_pred_label != label:
                #imshow(image.detach().cpu())
                #imshow(adversarial_sample.detach().cpu())
                generate_limit-=1

                # Save the adversarial images and labels
                image_path = os.path.join(valid_dir, f"adv_sample_validation_{idx}.png")
                vutils.save_image(image, image_path)
                valid_adv_samples.append({
                    "image_path": image_path,
                    "original_label": label.item(),
                    "adversarial_label": adversarial_pred_label.item(),
                })
                idx += 1
            else:
                print("FGSM failure -> skipping images")

            if generate_limit==0:
                break

def generate_adversarial_testing_samples(params, idx):
    train_loader, val_loader, test_loader, model, device = setup(params)
    loss_fn = params.get("loss_fn")
    # print the size of the test_loader
    print(f"Testing loader size: {len(test_loader.dataset)}")
    generate_limit = len(test_loader.dataset) // 10 # 10% of the dataset

    model.to(device)
    model.load_state_dict(torch.load(MODEL_FILEPATH))
    model.eval()
    print(f"Model loaded to device: {device}")

    renormalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #generate_limit = params.get("generate_limit", 0)

    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)

        image.requires_grad = True
        prediction = model.forward(image)
        pred_label = torch.argmax(torch.nn.functional.softmax(prediction, dim=1), dim=1)

        print(f"Model output: {pred_label} | Label: {label}")

        if pred_label != label:
            print("Incorrect prediction, skipping FGSM for this sample")
            continue
        else:
            loss = loss_fn(prediction, label)
            model.zero_grad()
            loss.backward()

            adversarial_sample = renormalize(
                fgsm(denormalize(image), params.get("epsilon"), image.grad.data)
            )
            adversarial_prediction = model.forward(adversarial_sample)
            adversarial_pred_label = torch.argmax(torch.nn.functional.softmax(adversarial_prediction, dim=1), dim=1)
            print(f"Adversarial output: {adversarial_pred_label}")

            # Show the image pre-fgsm and post-fgsm
            # Note that we only care about initially correctly predicted samples that are later incorrectly classified
            if adversarial_pred_label != label:
                #imshow(image.detach().cpu())
                #imshow(adversarial_sample.detach().cpu())
                generate_limit-=1

                # Save the adversarial images and labels
                image_path = os.path.join(test_dir, f"adv_sample_testing_{idx}.png")
                vutils.save_image(image, image_path)
                test_adv_samples.append({
                    "image_path": image_path,
                    "original_label": label.item(),
                    "adversarial_label": adversarial_pred_label.item(),
                })
                idx += 1
            else:
                print("FGSM failure -> skipping images")

            if generate_limit==0:
                break

if __name__ == '__main__':
    params ={
        "batch_size": 1, # Ensure each datapoint has gradients independent of each other
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam,
        "lr": 1e-3,
        "model_save_path": "models/resnet18_base/epoch_6.pth",
        "epsilon": 0.01,
        #"generate_limit": 400
    }
    generate_adversarial_training_samples(params,0)
    generate_adversarial_validation_samples(params,0)
    generate_adversarial_testing_samples(params,0)


# Save as CSV
pd.DataFrame(train_adv_samples).to_csv(os.path.join(train_dir, "train_adv_samples.csv"), index=False)
pd.DataFrame(valid_adv_samples).to_csv(os.path.join(valid_dir, "valid_adv_samples.csv"), index=False)
pd.DataFrame(test_adv_samples).to_csv(os.path.join(test_dir, "test_adv_samples.csv"), index=False)
