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

    model = vae_models[config['model_params']['name']](**config['model_params'])
    logdir = r"logs\Screentone-VAE\version_32"
    model_path = os.path.join(logdir, r"checkpoints\epoch=8-step=45755_2.ckpt")
    tmp_model = torch.load(model_path)
    vgg_state_dict = {}
    for key, value in tmp_model["state_dict"].items():
        if "model.vgg." in key:
            vgg_state_dict[key[10:]] = value
    model.vgg.load_state_dict(vgg_state_dict)
    
    experiment = VAEXperiment(None, config['exp_params'])
    dataloader = experiment.val_dataloader()

    logdir = r"logs\Vgg"
    writer = SummaryWriter(logdir)
    first = True
    count = 0
    total_z = None
    total_image = None
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    for image, label in dataloader:
        tmp = torch.cat([image, image, image], 1)
        z  = [GramMatrix()(A).detach() for A in vgg(tmp, style_layers)]
        print(z.shape)
        if first:
            total_z = np.array(z)
            total_image = image.cpu().detach().numpy()
            first = False
        else:
            total_z = np.concatenate((total_z, np.array(z)), axis=0)
            total_image = np.concatenate((total_image, image.cpu().detach().numpy()), axis=0)

        count += image.shape[0]
        if count > 3000:
            break
        print(count)
        break
    writer.add_embedding(torch.from_numpy(total_z), label_img=torch.from_numpy(total_image)) # metadata=np.array([0]*total_z.shape[0])
    writer.close()
