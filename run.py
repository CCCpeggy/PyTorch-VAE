import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

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


    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[config['model_params']['name']](**config['model_params'])
    if False: # load old model
        model_path = r"logs\Screentone-VAE\version_51\checkpoints\epoch=44-step=236384.ckpt"
        tmp_model = torch.load(model_path)
        model.load_state_dict(tmp_model)
    experiment = VAEXperiment(model,
                            config['exp_params'])

    runner = Trainer(weights_save_path=f"{tt_logger.save_dir}",
                    min_epochs=1,
                    logger=tt_logger,
                    flush_logs_every_n_steps=100,
                    limit_train_batches=1.,
                    limit_val_batches=1.,
                    num_sanity_val_steps=5,
                    **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
