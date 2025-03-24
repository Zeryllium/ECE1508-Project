import torch
from torchvision.transforms import transforms

from main import denormalize
from main import imshow
from main import setup

MODEL_FILEPATH = "models/resnet18_base/epoch_6.pth"

def fgsm(image, epsilon, gradient):
    modified_image = image + epsilon * gradient.sign()
    # Clamp the images to the same input normalization (between 0 and 1)
    return torch.clamp(modified_image, 0, 1)

def generate_adversarial_samples(params):
    train_loader, val_loader, test_loader, model, device = setup(params)
    loss_fn = params.get("loss_fn")

    model.to(device)
    model.load_state_dict(torch.load(MODEL_FILEPATH))
    model.eval()
    print(f"Model loaded to device: {device}")

    renormalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    generate_limit = params.get("generate_limit", 0)

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
                imshow(image.detach().cpu())
                imshow(adversarial_sample.detach().cpu())
                generate_limit-=1
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
        "generate_limit": 10
    }
    generate_adversarial_samples(params)
