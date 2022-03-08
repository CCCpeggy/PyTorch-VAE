from models import *
# import tensorboardX as tbx
from sklearn.decomposition import PCA
import os
import cv2
import yaml
import torch
import shutil
import argparse
import numpy as np

from sklearn.cluster import *

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

    # logdir = r"logs\Screentone-VAE\version_32"
    # model_path = os.path.join(logdir, r"checkpoints\epoch=8-step=45755_2.ckpt")
    logdir = r"logs\Screentone-VAE\version_39"
    model_path = os.path.join(logdir, r"checkpoints\epoch=79-step=331439.ckpt")
    global_path = "../../../shared/datasets/gabor_noise_band_5_10"
    tmp_model = torch.load(model_path)
    # # print(checkpoint)
    # model = VQVAE(1, 64, 512)
    # model.load_state_dict(checkpoint["state_dict"])


    model = vae_models[config['model_params']['name']](**config['model_params'])

    encoder_state_dict = {}
    fc_mu_state_dict = {}
    fc_var_state_dict = {}
    decoder_state_dict = {}
    decoder_input_state_dict = {}
    final_layer_state_dict = {}
    feature_network_state_dict = {}
    for key, value in tmp_model["state_dict"].items():
        if "model.encoder." in key:
            encoder_state_dict[key[14:]] = value
        elif "model.fc_mu." in key:
            fc_mu_state_dict[key[12:]] = value
        elif "model.fc_var." in key:
            fc_var_state_dict[key[13:]] = value
        elif "model.decoder." in key:
            decoder_state_dict[key[14:]] = value
        elif "model.decoder_input." in key:
            decoder_input_state_dict[key[20:]] = value
        elif "model.final_layer." in key:
            final_layer_state_dict[key[18:]] = value
        elif "model.feature_network." in key:
            feature_network_state_dict[key[22:]] = value
        else:
            print(key)

    model.encoder.load_state_dict(encoder_state_dict)
    model.fc_mu.load_state_dict(fc_mu_state_dict)
    model.fc_var.load_state_dict(fc_var_state_dict)
    # model.decoder.load_state_dict(decoder_state_dict)
    # model.decoder_input.load_state_dict(decoder_input_state_dict)
    if len(final_layer_state_dict.keys()) > 0:
        model.final_layer.load_state_dict(final_layer_state_dict)
    if len(feature_network_state_dict.keys()) > 0:
        model.feature_network.load_state_dict(feature_network_state_dict)

    experiment = VAEXperiment(model, config['exp_params'])
    dataloader = experiment.predict_dataloader()

    first = True
    total_z = None
    total_image = None
    total_path = []
    count = 0
    for image, path in dataloader:
        total_path += path
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
        print(count)

  
    pca = PCA(n_components=10, svd_solver='arpack')
    z_pca = pca.fit_transform(total_z)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print('L_sk.shape:', z_pca.shape)
    # print('L_sk:', z_pca)

    # db = DBSCAN(eps=9, min_samples=3).fit(z_pca)
    # db = AgglomerativeClustering(n_clusters=20).fit(z_pca)
    db = SpectralClustering(n_clusters=20).fit(z_pca)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    labels = [i+1 for i in labels]

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # print(labels)

    def makedir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    classify_dir = "classify3"
    cut_dir = "classify_cut_3"
    makedir(classify_dir)
    makedir(cut_dir)
    for i in set(labels):
        makedir(os.path.join(classify_dir, str(i)))
        makedir(os.path.join(cut_dir, str(i)))
    for i in range(count):
        shutil.copy(total_path[i], os.path.join(classify_dir, str(labels[i])))
        cv2.imwrite(os.path.join(cut_dir, str(labels[i]), "{}.png".format(i)), np.moveaxis(total_image[i], 0, -1)*255)
        
