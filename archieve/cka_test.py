import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os

def compute_and_save_cka_heatmap(model1, model2, dataloader1, dataloader2, save_path='cka_heatmap.png'):
    # Function to extract layer names from the model
    def get_layer_names(model):
        layer_names = []
        for name, module in model.named_modules():
            layer_names.append(name)
        return layer_names

    # Set the model to evaluation mode
    model1.eval()
    model2.eval()

    # Automatically get the layer names for both models
    layer_names_1 = get_layer_names(model1)
    layer_names_2 = get_layer_names(model2)

    # Initialize CKA with unique model names
    from torch_cka import CKA
    cka = CKA(
        model1=model1,
        model2=model2,
        model1_name="ResNet18",  # Provide unique names
        model2_name="ResNet34",
        model1_layers=layer_names_1,
        model2_layers=layer_names_2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Perform CKA comparison on the dataloader
    cka.compare(dataloader1=dataloader1, dataloader2=dataloader2)

    cka.plot_results(save_path=save_path)

    print(f"CKA heatmap saved to: {save_path}")

if __name__ == "__main__":
    # Load pre-trained ResNet18 and ResNet34 models
    resnet18 = models.resnet50(pretrained=True)
    resnet34 = models.resnet50(pretrained=True)

    # Define transforms for ImageNet-like dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load a larger dataset (using CIFAR-10 here for demo, replace with ImageNet if needed)
    dataset = datasets.CIFAR10(root='data', train=False, transform=transform, download=True)

    for batch in dataset:
        images, labels = batch
        print(images.shape)

    # Select a subset of the dataset (e.g., first 20 images)
    subset_indices = list(range(32))
    small_dataset = Subset(dataset, subset_indices)

    # Create DataLoader for the subset
    dataloader = DataLoader(small_dataset, batch_size=16, shuffle=False)

    # Check for NaNs in layer outputs
    def check_for_nan(module, input, output):
        if torch.isnan(output).any():
            print(f"NaN detected in module: {module}")

    # Register hooks to check for NaNs
    for name, module in resnet18.named_modules():
        module.register_forward_hook(check_for_nan)
    for name, module in resnet34.named_modules():
        module.register_forward_hook(check_for_nan)

    # Call the function to compute and save the CKA heatmap between two models
    compute_and_save_cka_heatmap(resnet18, resnet34, dataloader, dataloader, save_path='cka_heatmap_resnet18_vs_resnet34.png')