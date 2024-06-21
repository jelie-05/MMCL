import os
import pickle
import torch

from configs.networks.Siamese_Resnet import Lidar_Backbone
from configs.networks.Siamese_Resnet import Classifier_Lidar
from configs.networks.Siamese_Resnet import Image_Backbone


def load_model_lid(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = Lidar_Backbone(model_dict["hparams"])
    model.load_state_dict(model_dict["state_dict"])
    print("model is loaded")
    return model


def load_model_cls_lid(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = Classifier_Lidar(model_dict["hparams"])
    model.load_state_dict(model_dict["state_dict"])
    print("model is loaded")
    return model


def load_model_img(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["cifar_classification_ptl"]
    model = Image_Backbone(model_dict["hparams"])
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