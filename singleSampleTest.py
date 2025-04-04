import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as transforms

from resnet18 import ResNet18


def test_model_single_sample(model, image, label, device):
    model.eval()

    image = image.to(device)

    prediction = model.forward(image.unsqueeze(0)).squeeze()

    print(f"Confidence level per class: {torch.nn.functional.softmax(prediction, dim=0)}")
    print(f"Predicted label: {torch.argmax(prediction)} | True label: {label}")

    return

if __name__ == "__main__":
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    source_image = transforms(plt.imread("source_sample.png"))
    adversarial_image = transforms(plt.imread("adversarial_sample.png"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18().to(device)

    # Baseline
    model.load_state_dict(torch.load("models/resnet18_base/2025-04-03T16-37_epoch_4.pth"))

    print("Regular sample")
    test_model_single_sample(model, source_image, "8", device)
    print("Adversarial sample")
    test_model_single_sample(model, adversarial_image, "8", device)

    # Adversarial
    model.load_state_dict(torch.load("models/resnet18_adversarial/2025-04-03T17-43_epoch_3.pth"))

    print("Regular sample")
    test_model_single_sample(model, source_image, "8", device)
    print("Adversarial sample")
    test_model_single_sample(model, adversarial_image, "8", device)


    # Baseline + Augmented
    model.load_state_dict(torch.load("models/resnet18_base_augmented/2025-04-03T18-28_epoch_3.pth"))

    print("Regular sample")
    test_model_single_sample(model, source_image, "8", device)
    print("Adversarial sample")
    test_model_single_sample(model, adversarial_image, "8", device)

    # Adversarial + Augmented
    model.load_state_dict(torch.load("models/resnet18_adversarial_augmented/2025-04-03T18-53_epoch_5.pth"))

    print("Regular sample")
    test_model_single_sample(model, source_image, "8", device)
    print("Adversarial sample")
    test_model_single_sample(model, adversarial_image, "8", device)
