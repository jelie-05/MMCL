import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from utils import add_colorbar
import os


class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader) kitti_dataloader for model1
        :param dataloader2: (DataLoader) If given, model2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("kitti_dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        # Initialize HSIC matrix to store results
        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader2))

        for x1_batch, x2_batch in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
            self.model1_features = {}
            self.model2_features = {}

            # Ensure x1_batch and x2_batch are moved to the device
            x1_batch = x1_batch.to(self.device)
            x2_batch = x2_batch.to(self.device)

            # Forward pass through both models
            _ = self.model1(x1_batch)
            _ = self.model2(x2_batch)

            # Loop over features extracted from model1
            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)  # Flattening along all except batch dimension
                K = X @ X.t()  # Compute Gram matrix for model1 features
                K.fill_diagonal_(0.0)  # Set diagonal to zero
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                # Loop over features extracted from model2
                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)  # Flatten model2 features
                    L = Y @ Y.t()  # Compute Gram matrix for model2 features
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mismatch! {K.shape}, {L.shape}"

                    # Compute HSIC between model1 and model2 features
                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches
                    assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

        # Normalize the HSIC matrix for final CKA calculation
        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None,
                     show_plot=False,
                     vmin=0,  # Minimum limit for color scale
                     vmax=1):  # Maximum limit for color scale
        fig, ax = plt.subplots()

        # Set the color limits (vmin and vmax) for consistency across plots
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)

        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=20)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=20)

        # Set the font size for the axis tick labels
        ax.tick_params(axis='both', which='major', labelsize=17)  # Adjust the label size as needed

        # if title is not None:
        #     ax.set_title(f"{title}", fontsize=25)
        # else:
        #     ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=25)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"Directory {save_dir} created successfully.")
            else:
                print(f"Directory {save_dir} already exists.")

            plt.savefig(save_path, dpi=300)
            print(f"Plot successfully saved to {save_path}")

        if show_plot:
            plt.show()
