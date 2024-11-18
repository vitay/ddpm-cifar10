import os, sys
import logging

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

# Device
def get_device():
    "Returns the available device."

    if torch.cuda.is_available(): # GPU
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # Metal (Macos)
        device = torch.device("mps")
    else: # CPU
        device = torch.device("cpu")

    return device

# Plotting
def plot_images(images):
    "Plot images."
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    "Save images as an image"
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

# Directory management
def create_directory(directory_list):
    "Creates the directories if they do not exist."
    for path in directory_list:
        if not os.path.exists(path):
            logging.info(f"Creating directory: {path}")
            os.makedirs(path)