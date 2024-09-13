import os.path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from src.datasets.kitti_loader.dataset_2D import create_dataloaders
import multiprocessing
# from torch_cka import CKA
from cka_mod import CKA
def compute_and_save_cka_heatmap(model1, model2, dataloader1, dataloader2, save_path='cka_heatmap.png', show_plot=False):
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

    cka = CKA(
        model1=model1,
        model2=model2,
        model1_name="Encoder Image",  # Provide unique names
        model2_name="Encoder Lidar",
        model1_layers=layer_names_1,
        model2_layers=layer_names_2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Perform CKA comparison on the dataloader
    cka.compare(dataloader1=dataloader1, dataloader2=dataloader2)

    cka.plot_results(save_path=save_path, show_plot=show_plot)

    print(f"CKA heatmap saved to: {save_path}")

def cka_analysis(data_root, output_dir, model_im, model_lid, perturbation_eval, show_plot=False):

    batch_size = 64
    num_cores = min(multiprocessing.cpu_count(), 64)
    dataloader_im, dataloader_lid = create_dataloaders(root=data_root, perturb_filenames=perturbation_eval, mode='check',
                                                       batch_size=batch_size, num_cores=num_cores)

    # Check for NaNs in layer outputs
    def check_for_nan(module, input, output):
        if torch.isnan(output).any():
            print(f"NaN detected in module: {module}")

    # Register hooks to check for NaNs
    for name, module in model_im.named_modules():
        module.register_forward_hook(check_for_nan)
    for name, module in model_lid.named_modules():
        module.register_forward_hook(check_for_nan)

    save_path = os.path.join(output_dir, 'cka_analysis.png')
    # Call the function to compute and save the CKA heatmap between two models
    compute_and_save_cka_heatmap(model_im, model_lid, dataloader_im, dataloader_lid, save_path=save_path, show_plot=show_plot)
