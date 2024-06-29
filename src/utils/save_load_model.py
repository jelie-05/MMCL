import os
import pickle
import torch

from src.models.mm_siamese import lidar_backbone
from src.models.mm_siamese import image_backbone
from src.models.classifier_head import classifier_lidar


def load_model_lid(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = lidar_backbone(model_dict["hparams"])
    model.load_state_dict(model_dict["state_dict"])
    print("model is loaded")
    return model


def load_model_cls_lid(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = classifier_lidar(model_dict["hparams"])
    model.load_state_dict(model_dict["state_dict"])
    print("model is loaded")
    return model


def load_model_img(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = image_backbone(model_dict["hparams"])
    model.load_state_dict(model_dict["state_dict"])
    print("model is loaded")
    return model


def save_model(model, file_name, directory="models"):

    model = model.cpu()
    model_dict = {"cifar_classification_ptl": {"state_dict": model.state_dict(), "hparams": model.hp}}
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("dir created")
    pickle.dump(model_dict, open(os.path.join(directory, file_name), 'wb', 4))
    print("Model's parameters saved")