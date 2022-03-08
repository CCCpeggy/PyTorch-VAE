from models import *
# import tensorboardX as tbx
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import yaml
import torch
import argparse
import numpy as np

from models import VanillaVAE
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

    logdir = r"logs\Screentone-VAE\version_32"
    model_path = os.path.join(logdir, r"checkpoints\epoch=8-step=45755_2.ckpt")
    tmp_model = torch.load(model_path)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.load_state_dict(tmp_model)

    experiment = VAEXperiment(model, config['exp_params'])
    dataloader = experiment.predict_dataloader()

    writer = SummaryWriter(logdir)
    first = True
    count = 0
    total_z = None
    total_image = None
    total_label = []
    for image, path in dataloader:
        total_label += [os.path.normpath(p).split(os.sep)[-2] for p in path]
        mu, log_var = model.encode(image)
        z = model.reparameterize(mu, log_var)
        if first:
            total_z = z.cpu().detach().numpy()
            total_image = image.cpu().detach().numpy()
            first = False
        else:
            total_z = np.concatenate((total_z, z.cpu().detach().numpy()), axis=0)
            total_image = np.concatenate((total_image, image.cpu().detach().numpy()), axis=0)

        count += image.shape[0]
        if count > 3000:
            break
        print(count)
    writer.add_embedding(torch.from_numpy(total_z), global_step=2, label_img=torch.from_numpy(total_image), tag='conan', metadata=total_label)
    writer.close()
