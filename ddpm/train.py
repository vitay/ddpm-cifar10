import os, sys
import argparse
import yaml
import logging
import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from datasets import get_cifar10
from cifar_test import train_CNN
from model import UNet, EMA
from utils import get_device, save_images, create_directory, load_config

# Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Original DDPM model
class DiffusionModel:

    def __init__(self, config, device, img_size=256):
        
        # Parameters
        self.config = config
        self.noise_steps = config['noise_steps']
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.img_size = img_size
        self.device = device

        # Noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # UNet model
        self.model = UNet().to(device)

        # Exponentially moving average
        self.ema = EMA(self.model, beta=config['beta_ema'], device=device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.MSELoss()

    def prepare_noise_schedule(self):
        "Linear noise schedule between beta_start and beta_end over T steps."

        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Adds noise to the image x corresponding to the schedule at time t.

        $$
            x_t = \sqrt{\bar{\alpha}_t} \, x_0 +  \sqrt{1 - \bar{\alpha}_t} \, \epsilon_t \;\; \text{where} \; \epsilon_t \sim \mathcal{N}(0, I)
        $$
        """

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]

        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        epsilon = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        "Return n time steps sampled between 0 and T."
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, n):
        "Returns $n$ samples using the EMA model"
        logging.info(f"Sampling {n} new images....")
        
        self.model.eval()

        with torch.no_grad():

            # Input noise
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        
            # Iterate backwards over time
            for i in tqdm.tqdm(reversed(range(1, self.noise_steps)), position=0):
        
                # Time tensor
                t = (torch.ones(n) * i).long().to(self.device)
        
                # Predict noise
                predicted_noise = self.ema.model(x, t)
        
                # Noise schedule
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

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


    def train(self, train_loader):
        logging.info("Training the diffusion model")

        for epoch in range(config['num_epochs']):
            logging.info(f"Starting epoch {epoch}:")
            running_mse = 0.0
            num_samples = 0

            pbar = tqdm.tqdm(train_loader)
            for i, (images, _) in enumerate(pbar):
                # Get images
                images = images.to(self.device)

                # Get the time step
                t = self.sample_timesteps(images.shape[0]).to(self.device)

                # Create the target
                x_t, noise = self.noise_images(images, t)

                # Predict the noise
                predicted_noise = self.model(x_t, t)

                # Compute loss
                loss = self.criterion(noise, predicted_noise)

                # Optimizer logic
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # EMA update
                self.ema.step(self.model)

                # Metrics
                mse = loss.item()
                running_mse += mse
                num_samples += images.shape[0]
                pbar.set_postfix(MSE=running_mse/num_samples)

            # Epoch mse
            mse_epoch = running_mse/num_samples
            logging.info(f"MSE: {mse_epoch}:")

            # Validation
            sampled_images = self.sample(n=config['batch_size'])
            filename = os.path.join("results", f"{epoch}.jpg")
            save_images(sampled_images,filename)

            # Log
            if not config['no_wandb']: wandb.log({"mse": mse_epoch, "generation": wandb.Image(filename)})

            # Save checkpoint
            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"ddpm-ckpt.pt"))

# Main training loop
def main(config):

    # Wandb logging
    if not config['no_wandb']: run = wandb.init(project="DDPM-CIFAR10", config=config)

    # Select hardware: 
    device = get_device()
    logging.info(f"Device found: {device}")

    # Make sure the necessary directories exist
    create_directory(["./data", "./checkpoints", "./results"])

    # Get the data
    logging.info("Downloading the CIFAR10 dataset")
    train_data, train_loader = get_cifar10(batch_size=config['batch_size'], testset=False)
    logging.info(f"Training set: {len(train_data)} samples, input shape {train_data[0][0].shape}")

    # Create diffusion model
    diffusion = DiffusionModel(config=config, img_size=32, device=device)

    # Initial generation
    sampled_images = diffusion.sample(n=config['batch_size'])
    filename = os.path.join("results", f"initial.jpg")
    save_images(sampled_images, filename)
    if not config['no_wandb']: wandb.log({"generation": wandb.Image(filename)})

    # Train the diffusion model
    diffusion.train(train_loader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train DDPM on the dataset.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file (default: config.yaml)')
    parser.add_argument('--no-wandb', action="store_true", help="Whether to log with Weights and Biases (offline mode).")
    parser.add_argument('--batch_size', type=int, help="Batch size.")
    parser.add_argument('--num_epochs', type=int, help="Number of epochs.")

    # Parse arguments
    args = parser.parse_args()
    config = load_config(args)
    logging.info(config)

    # Main loop
    main(config)
