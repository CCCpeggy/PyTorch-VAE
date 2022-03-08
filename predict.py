import os
import cv2
import yaml
import torch
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment


if "__main__" == __name__:

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')
    
    args = parser.parse_args()
    with open(args.filename, 'r', encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("error", exc)

    model_path = r"logs\Screentone-VAE\version_41\checkpoints\epoch=51-step=219127.ckpt"
    tmp_model = torch.load(model_path)
    # # print(checkpoint)
    # model = VQVAE(1, 64, 512)
    # model.load_state_dict(checkpoint["state_dict"])

    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.load_state_dict(tmp_model)

    fake_path = r"output\fake3"
    real_path = r"output\real3"

    experiment = VAEXperiment(model, config['exp_params'])
    dataloader = experiment.val_dataloader()
    count = 1

    # reconstruction
    for image, label in dataloader:
        mu, log_var = model.encode(image)
        z = model.reparameterize(mu, log_var)
        recon_image = model.decode(z)
        image = np.array(image.detach())
        recon_image = np.array(recon_image.detach())
        for i in range(image.shape[0]):
            cv2.imwrite(os.path.join(real_path, str(count) + ".png"), image[i][0] * 255)
            cv2.imwrite(os.path.join(fake_path, str(count) + ".png"), recon_image[i][0] * 255)
            count += 1
        break

    # sample
    num_samples = 144
    z = torch.randn(num_samples, config['model_params']['latent_dim'])
    sample_image = model.decode(z)
    sample_image = np.array(sample_image.detach())
    for img in sample_image:
        cv2.imwrite(os.path.join(fake_path, str(count) + ".png"), img[0] * 255)
        count += 1