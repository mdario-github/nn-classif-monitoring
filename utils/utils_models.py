import gdown
import os
import torch

import models
from params.params_datasets import *
from params.params_models import *


def download_from_gdrive(gdrive_id, destination):
    url = gdrive_id  #"https://drive.google.com/uc?id=" + gdrive_id
    fod = destination
    gdown.download(url, fod, quiet=False)


def load_model(model_name, dataset_name, DEVICE=None):
    name_of_model = f"{model_name}_{dataset_name}.pth"
    path_to_model = os.path.join(path_to_saved_models, name_of_model)

    if not os.path.exists(path_to_model):
        download_from_gdrive(models_gdrive_ids[name_of_model], path_to_model)

    match model_name:
        case "resnet":
            model = _load_resnet(path_to_model, dataset_name, DEVICE)
        case "densenet":
            model = _load_densenet(path_to_model, dataset_name, DEVICE)
        case _:
            raise ValueError("Specified 'model_name' is not within ... []")
    return model


def _load_resnet(path_to_model, dataset_name, DEVICE):
    """Load the ResNet model.
    """
    model = models.ResNet34(num_c=n_classes_dataset[dataset_name])
    model.load_state_dict(torch.load(path_to_model, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model


def _load_densenet(path_to_model, dataset_name, DEVICE):
    """Load the DenseNet model.
    """
    model = models.DenseNet3(100, n_classes_dataset[dataset_name])
    model.load_state_dict(torch.load(path_to_model, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model