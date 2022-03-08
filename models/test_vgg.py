import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from .vgg import VGG, GramMatrix, GramMSELoss

imgsize = 32
dimtimes = imgsize // 32
img_channel = 1

class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * dimtimes * dimtimes, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * dimtimes * dimtimes, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * dimtimes * dimtimes)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=img_channel,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        # self.content_layers = ['r42']
        self.loss_layers = self.style_layers
        self.loss_fns = [GramMSELoss()] * len(self.style_layers)
        if torch.cuda.is_available():
            self.loss_fns = [loss_fn.cuda() for loss_fn in self.loss_fns]
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load('./models/vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        # self.content_weights = [1e0]
        self.weights = self.style_weights

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, dimtimes, dimtimes)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        input = torch.cat([input, input, input], 1)
        style_targets = [GramMatrix()(A).detach() for A in self.vgg(input, self.style_layers)]
        targets = style_targets
        recons = torch.cat([recons, recons, recons], 1)
        out = self.vgg(recons, self.loss_layers)
        layer_losses = [self.weights[a] * self.loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        recons_loss = sum(layer_losses) #F.mse_loss(recons, input)

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def load_state_dict(self, model):
        encoder_state_dict = {}
        fc_mu_state_dict = {}
        fc_var_state_dict = {}
        decoder_state_dict = {}
        decoder_input_state_dict = {}
        final_layer_state_dict = {}
        vgg_state_dict = {}
        for key, value in model["state_dict"].items():
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
            elif "model.vgg." in key:
                vgg_state_dict[key[10:]] = value
            else:
                print(key)

        self.encoder.load_state_dict(encoder_state_dict)
        self.fc_mu.load_state_dict(fc_mu_state_dict)
        self.fc_var.load_state_dict(fc_var_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        self.decoder_input.load_state_dict(decoder_input_state_dict)
        self.final_layer.load_state_dict(final_layer_state_dict)
        self.vgg.load_state_dict(vgg_state_dict)