# import tensorboardX as tbx
from turtle import forward
from sklearn.decomposition import PCA
import os
import cv2
import yaml
from tqdm import tqdm
import torch
from torch import optim
import shutil
import argparse
import numpy as np

from sklearn.cluster import *
from sklearn.neighbors import KDTree, NearestNeighbors

from models import *
from experiment import VAEXperiment

# in -> in 8 層，in -> out 8 層
class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        modules = []
        for _ in range(8):
            modules.append(
                nn.Sequential(nn.Linear(in_channel, in_channel))
            )
        modules.append(
            nn.Sequential(nn.Linear(in_channel, out_channel))
        )
        self.fc = nn.Sequential(*modules)
        self.loss_mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, x):
        return self.fc(x)
    
    def loss(self, real, fake):
        return self.loss_mse(real, fake)

# 解析路徑中的參數至 tensor
def parse_path(path):
    X = np.zeros((len(path), 6))
    axis0_idx = 0
    for p in path:
        p = os.path.basename(p)[:-4]
        x = [float(s) for s in p.split("_")]
        x[0] /= 200
        x[1] /= 40
        x[2] /= 20
        x[3] /= 3.14
        x[4] /= 10
        x[5] /= 12
        X[axis0_idx] = np.array(x)
        axis0_idx += 1
    return torch.FloatTensor(X)

if "__main__" == __name__:

    parser = argparse.ArgumentParser(description='學習編碼對應的參數')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'vae 的 config 檔案路徑',
                        default='configs/vae.yaml')
    
    args = parser.parse_args()
    with open(args.filename, 'r', encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("error", exc)


    # 載入 VAE 的 Encode 模型
    # logdir = r"logs\Screentone-VAE\version_32"
    # model_path = os.path.join(logdir, r"checkpoints\epoch=8-step=45755_2.ckpt")
    logdir = r"logs\Screentone-VAE\version_54"
    model_path = os.path.join(logdir, r"checkpoints\epoch=47-step=258863.ckpt")
    global_path = r"D:\shared\datasets\20220214\gabor_noise_band_5_10"
    # global_path = r"D:\shared\datasets\tmp"

    encoder = vae_models[config['model_params']['name']](**config['model_params'])
    encoder.load_state_dict(torch.load(model_path))

    experiment = VAEXperiment(encoder, config['exp_params'], select_rect=True)
    dataloader = experiment.predict_dataloader()

    # 取得 config
    global_exp_params = config['exp_params']
    global_exp_params['data_path'] = global_path
    global_experiment = VAEXperiment(encoder, global_exp_params, select_rect=False)
    global_dataloader = global_experiment.predict_dataloader()
   
    # create decoder
    decoder = MLP(config['model_params']["latent_dim"], 6)
    optimizer = optim.Adam(
        decoder.parameters(), lr=global_exp_params['LR'], 
        weight_decay=global_exp_params['weight_decay']
    )

    for epoch in range(100000):
        print(f"epoch {epoch}")
        for image, path in tqdm(global_dataloader):
            
            # get fake parameter
            mu, log_var = encoder.encode(image)
            z = encoder.reparameterize(mu, log_var)
            fake = decoder(z)
            
            # get real parameter
            real = parse_path(path)
            
            loss = decoder.loss(real, fake)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"loss: {loss}")
        torch.save(decoder.state_dict(), 
            f"./{config['logging_params']['save_dir']}ParamDecoder/epoch-{epoch}.ckpt")
        