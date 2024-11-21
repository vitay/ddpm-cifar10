import os, sys
import argparse
import logging
import tqdm

import torch

from schedule import DiffusionSchedule, LinearSchedule
from utils import get_device, save_images, load_config

# Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def sample(model: "UNet", schedule:DiffusionSchedule, n:int, img_size:int, T:int, device:str) -> torch.tensor:
    """
    Returns $n$ samples using the model.

    Parameters:
    * model: UNet model used for the prediction.
    * schedule: schedule for alpha and beta
    * n: number of samples to return
    * img_size: size of the image (e.g. 32)

    """
    logging.info(f"Sampling {n} new images....")
    
    model.eval()

    with torch.no_grad():

        # Input noise
        x = torch.randn((n, 3, img_size, img_size)).to(device)
    
        # Iterate backwards over time
        for i in tqdm.tqdm(reversed(range(1, T)), position=0):
    
            # Time tensor
            t = (torch.ones(n) * i).long().to(device)
    
            # Predict noise
            predicted_noise = model(x, t)
    
            # Noise schedule
            alpha = schedule.alpha[t][:, None, None, None]
            alpha_hat = schedule.alpha_hat[t][:, None, None, None]
            beta = schedule.beta[t][:, None, None, None]

            # Last step does not get noise
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # Interpolation
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    
    # Clamping to get a real image
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)

    return x

def test_model(config):
    device = get_device()

    # Create model and schedule
    model = torch.load(config['model'])
    schedule = LinearSchedule(beta_start=config['beta_start'], beta_end=config['beta_end'], T=config['noise_steps'], device=device)

    # Main loop
    samples = sample(model, schedule, config.n, config.img_size, config.noise_steps, device)

    # Make the plot
    save_images(samples, "results/samples.jpg")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate images using the saved model")
    parser.add_argument('--model', type=str, default="checkpoints/emamodel-checkpoint.pt", help="Name of the file.")
    parser.add_argument('--n', type=int, default=64, help="Number of images to generate.")

    # Parse arguments
    args = parser.parse_args()
    config = load_config(args)
    logging.info(config)

    test_model(config)

