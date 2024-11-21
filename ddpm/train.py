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
from utils import get_device, save_images, create_directory, load_config
from model import UNet, EMA
from schedule import DiffusionSchedule, LinearSchedule
from inference import sample

# Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Original DDPM model
class DiffusionModel:

    def __init__(self, config, device, img_size=256):
        
        # Parameters
        self.config = config
        self.noise_steps = config['noise_steps']
        self.img_size = img_size
        self.device = device

        # Noise schedule
        self.schedule = LinearSchedule(config['beta_start'], config['beta_end'], config['noise_steps'], self.device)

        # UNet model
        self.model = UNet().to(device)

        # Exponentially moving average
        if config['use_ema']:
            self.ema = EMA(self.model, beta=config['beta_ema'], device=device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.MSELoss()


    def noise_images(self, x, t):
        r"""
        Adds noise to the image x corresponding to the schedule at time t.

        $$
            x_t = \sqrt{\bar{\alpha}_t} \, x_0 +  \sqrt{1 - \bar{\alpha}_t} \, \epsilon_t \;\; \text{where} \; \epsilon_t \sim \mathcal{N}(0, I)
        $$
        """

        sqrt_alpha_hat = torch.sqrt(self.schedule.alpha_hat[t])[:, None, None, None]

        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.schedule.alpha_hat[t])[:, None, None, None]

        epsilon = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon



    def train(self, train_loader):
        logging.info("Training the diffusion model")

        for epoch in range(config['num_epochs']):
            logging.info(f"Starting epoch {epoch}:")
            running_mse = 0.0
            num_samples = 0

            self.model.train()

            pbar = tqdm.tqdm(train_loader)
            for i, (images, _) in enumerate(pbar):

                # Get images
                images = images.to(self.device)

                # Get the time step
                t = self.schedule.sample(images.shape[0]).to(self.device)

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
                if config['use_ema']: self.ema.step(self.model)

                # Metrics
                mse = loss.item()
                running_mse += mse
                num_samples += images.shape[0]
                pbar.set_postfix(MSE=running_mse/num_samples)

            # Epoch mse
            mse_epoch = running_mse/num_samples
            logging.info(f"MSE: {mse_epoch}:")

            # Validation
            model = self.ema_model if config['use_ema'] else self.model
            sampled_images = sample(model=model, schedule=self.schedule, n=config['batch_size'], img_size=config['img_size'], T=config['noise_steps'], device=self.device)

            # Save the images
            filename = f"results/{epoch}.jpg"
            save_images(sampled_images, filename)

            # Log
            if not config['no_wandb']: wandb.log({"mse": mse_epoch, "generation": wandb.Image(filename)})

            # Save checkpoint
            torch.save(self.model.state_dict(), config['trained-checkpoint'])
            torch.save(self.ema.model.state_dict(), config['ema-checkpoint'])

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
    logging.info("Create the UNet model")
    diffusion = DiffusionModel(config=config, img_size=config['img_size'], device=device)

    # Initial generation
    logging.info("Initial sampling")
    sampled_images = sample(model=diffusion.model, schedule=diffusion.schedule, n=config['batch_size'], img_size=config['img_size'], T=config['noise_steps'], device=device)

    # Save the image
    filename = f"results/initial.jpg"
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
