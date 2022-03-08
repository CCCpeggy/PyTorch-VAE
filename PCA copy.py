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
from sklearn.neighbors import KDTree, NearestNeighbors

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
    logdir = r"logs\Screentone-VAE\version_54"
    model_path = os.path.join(logdir, r"checkpoints\epoch=47-step=258863.ckpt")
    global_path = "../../../shared/datasets/basic"
    tmp_model = torch.load(model_path)
    # # print(checkpoint)
    # model = VQVAE(1, 64, 512)
    # model.load_state_dict(checkpoint["state_dict"])

    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.load_state_dict(tmp_model)

    experiment = VAEXperiment(model, config['exp_params'])
    dataloader = experiment.predict_dataloader()

    global_exp_params = config['exp_params']
    global_exp_params['data_path'] = global_path
    global_experiment = VAEXperiment(model, global_exp_params)
    global_dataloader = global_experiment.predict_dataloader()
    first = True
    global_z = None
    global_image = None
    global_file_name = []
    global_count = 0
    for image, path in global_dataloader:
        mu, log_var = model.encode(image)
        z = model.reparameterize(mu, log_var)
        if first:
            global_z = z.cpu().detach().numpy()
            global_image = image.cpu().detach().numpy()
            first = False
        else:
            global_z = np.concatenate((global_z, z.cpu().detach().numpy()), axis=0)
            global_image = np.concatenate((global_image, image.cpu().detach().numpy()), axis=0)
        global_file_name.append(path)
        global_count += image.shape[0]
        if global_count > 1000:
            break
        print(global_count)

    pca = PCA(n_components=10, svd_solver='arpack')
    global_pca_z = pca.fit_transform(global_z)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print('L_sk.shape:', global_pca_z.shape)

    first = True
    local_z = None
    local_image = None
    local_path = []
    local_count = 0
    for image, path in dataloader:
        local_path += path
        mu, log_var = model.encode(image)
        z = model.reparameterize(mu, log_var)
        if first:
            local_z = z.cpu().detach().numpy()
            local_image = image.cpu().detach().numpy()
            first = False
        else:
            local_z = np.concatenate((local_z, z.cpu().detach().numpy()), axis=0)
            local_image = np.concatenate((local_image, image.cpu().detach().numpy()), axis=0)
        local_count += image.shape[0]
        print(local_count)

    local_pca_z = pca.transform(local_z)

    # db = DBSCAN(eps=9, min_samples=3).fit(local_pca_z)
    # db = AgglomerativeClustering(n_clusters=400).fit(np.concatenate((local_pca_z, global_pca_z), axis=0))
    db = AgglomerativeClustering(n_clusters=18).fit(local_pca_z)
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
    
    classify_dir = "classify"
    classify_dir = "../../Dataset/Batch/Conan_78_7/sc_classify_auto"
    # cut_dir = "classify_cut"
    # global_dir = "classify_global_3"
    makedir(classify_dir)
    # makedir(cut_dir)
    label_all = set(labels)
    for i in label_all:
        makedir(os.path.join(classify_dir, str(i)))
        # makedir(os.path.join(cut_dir, str(i)))
        # makedir(os.path.join(global_dir, str(i)))
    tree = NearestNeighbors(n_neighbors=1).fit(global_pca_z)

    maximum_size = max([len([1 for i in range(local_count) if l == labels[i]]) for l in label_all])
    size = config['exp_params']['img_size']
    classify_image = np.zeros(((max(label_all)+1) * size, (maximum_size+1)* size))
    
    for l in label_all:
        min_dist = 5
        min_ind = 0
        sum_z = np.zeros(local_pca_z.shape[1])
        count = 0
        for i in range(local_count):
            if l == labels[i]:
                shutil.copy(local_path[i], os.path.join(classify_dir, str(labels[i])))
                image = np.moveaxis(local_image[i], 0, -1) * 255
                cv2.imwrite(os.path.join(cut_dir, str(labels[i]), f"local_{i}.png"), image)
                sum_z += local_pca_z[i]
                # print(classify_image.shape, l*size, (l+1)*size, classify_image[l*size:(l+1)*size, count*size:(count+1)*size].shape, image[:,:,0].shape)
                classify_image[l*size:(l+1)*size, (count+1)*size:(count+2)*size] = image[:,:,0]
                count += 1
                # dist, ind = tree.kneighbors(local_pca_z[i:i+1])
                # dist, ind = dist[0][0], ind[0][0]
                # if dist < min_dist:
                #     min_dist = dist
                #     min_ind = ind
        shutil.copy(os.path.join(classify_dir, str(labels[i])), os.path.join(classify_dir, global_file_name[idx]))
        dist, ind = tree.kneighbors([sum_z / count])
        dist, ind = dist[0][0], ind[0][0]
        image = np.moveaxis(global_image[ind], 0, -1) * 255
        classify_image[l*size:(l+1)*size, 0:size] = image[:,:,0]
        # global_image_path = os.path.join(cut_dir, str(l), f"global_{dist}.png")
        # cv2.imwrite(global_image_path, image)
    cv2.imwrite(os.path.join(classify_dir, "classify_image.png"), classify_image)
    
    
        # if min_dist < 5:
        #     global_image_path = os.path.join(cut_dir, str(labels[i]), f"global_{i}_{min_dist}.png")
        #     image = np.moveaxis(global_image[min_ind], 0, -1) * 255
        #     cv2.imwrite(global_image_path, image)
    # for i in range(global_count):
    #     ii = local_count + i
    #     cv2.imwrite(os.path.join(cut_dir, str(labels[ii]), "global_{}.png".format(i)), np.moveaxis(global_image[i], 0, -1)*255)

# regular vs unregular
