import os
import cv2
import yaml
import torch
from torch import nn
import argparse
import numpy as np
import random
from torchvision import transforms

from models import ScreenVAE
from experiment import VAEXperiment

import sys
sys.path.append('../procedural-advml')
from utils_noise import PlotGaborAni

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
    
    # def load_state_dict(self, model):
    #     encoder_state_dict = {}
    #     for key, value in model["state_dict"].items():
    #         if "model.encoder." in key:
    #             encoder_state_dict[key[14:]] = value
    #         else:
    #             print(key)

    #     self.encoder.load_state_dict(encoder_state_dict)

def param(x):
    p = {}
    p['num_kern'] = int(x[0] * 200)
    p['ksize'] = int(x[1] * 40)
    p['sigma'] = float(x[2] * 20)
    p['theta'] = float(x[3] * 3.14)
    p['lambd'] = float(x[4] * 10)
    p['sides'] = int(x[5] * 12)
    return p

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
    encode_path = os.path.join(r"logs\Screentone-VAE\version_54", r"checkpoints\epoch=47-step=258863.ckpt")
    decode_path = os.path.join(r"logs\ParamDecoder", r"epoch-4.ckpt")
    global_path = r"D:\shared\datasets\20220214\gabor_noise_band_5_10"
    # global_path = r"D:\shared\datasets\tmp"

    # encoder = vae_models[config['model_params']['name']](**config['model_params'])
    encoder = ScreenVAE(**config['model_params'])
    print("Load encoder")
    encoder.load_state_dict(torch.load(encode_path))
    print("Load encoder ... Done")

    experiment = VAEXperiment(encoder, config['exp_params'], select_rect=True)
    dataloader = experiment.predict_dataloader()

    # 取得 config
    global_exp_params = config['exp_params']
    global_exp_params['data_path'] = global_path
    global_experiment = VAEXperiment(encoder, global_exp_params, select_rect=False)
    global_dataloader = global_experiment.predict_dataloader()
   
    # create decoder
    decoder = MLP(config['model_params']["latent_dim"], 6)
    print("Load decoder")
    decoder.load_state_dict(torch.load(decode_path))
    print("Load decoder ... Done")

    output_path = f"./{config['logging_params']['save_dir']}ParamDecoder"
    
    test_len = 5
    image = []
    real = []
    for i in range(test_len):
        print(i)
        s_num_kern = random.randint(5, 200)
        s_ksize = random.randint(3, 40)
        s_sigma = random.uniform(2, 20)
        s_theta = random.uniform(0, np.pi)
        s_lambd = random.uniform(5, 10)
        s_sides = random.randint(1, 12)
        img = PlotGaborAni(s_num_kern, s_ksize, s_sigma, s_theta, s_lambd, s_sides)
       
        real.append([s_num_kern / 200, s_ksize / 40, s_sigma / 20, s_theta / 3.14, s_lambd / 10, s_sides / 12])
        
        img = transforms.ToTensor()(img[:32, :32])
        image.append(img)
    image = torch.stack(image, 0)

    # get fake parameter
    mu, log_var = encoder.encode(image.float())
    z = encoder.reparameterize(mu, log_var)
    fake = decoder(z)
    
    # get real parameter
    real = torch.FloatTensor(real)
    
    loss = decoder.loss(real, fake)
        
    fake = fake.cpu().detach().numpy()
    real = real.cpu().detach().numpy()

    imgs = np.zeros((test_len * 128, 2 * 128), int)
    for i in range(test_len):
        try:
            real_img = PlotGaborAni(**param(real[i, :]))
            fake_img = PlotGaborAni(**param(fake[i, :]))
            imgs[i*128:(i+1)*128, :128] = real_img
            imgs[i*128:(i+1)*128, 128:] = fake_img
        except:
            print(f"error {fake[i, :]}, {real[i, :]}")
    cv2.imwrite(f"{output_path}/re-generate.png", imgs)
    # break