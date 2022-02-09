import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from typing import List, Tuple
from collections import OrderedDict

from convblocks import Conv2dBlock, ConvTranspose2dBlock


# ENCODER

class Encoder(nn.Module):

    def __init__(self,
                input_shape: List[int],
                conv_filters:List[int], # must have 1 more element than others -> FIRST element must be 3 for colored images
                conv_kernels: List[Tuple[int]],
                conv_strides: List[Tuple[int]],
                paddings: List[Tuple[int]],
                dilations: List[Tuple[int]],
                latent_space_dim: int,
                **kwargs):
    
        super(Encoder, self).__init__(**kwargs)
        
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels 
        self.conv_strides = conv_strides
        self.paddings = paddings
        self.dilations = dilations
        self.latent_space_dim = latent_space_dim

        
        # dim assertion

        assert len(self.conv_kernels) == len(self.conv_strides) == len(self.paddings)

        self.convblocks = nn.Sequential(
            OrderedDict(
                [
            (
            f"Convolution_Block_{i+1}",
            Conv2dBlock(conv_filters_in=self.conv_filters[i],
                        conv_filters_out=self.conv_filters[i+1],
                        conv_kernels=self.conv_kernels[i],
                        conv_strides=self.conv_strides[i],
                        paddings=self.paddings[i],
                        dilations=self.dilations[i]).float()
            )
            
            for i in range(len(self.conv_filters) - 1)
            
                ]
            )
        )   

        self.shape_before_bottleneck = self._calculate_shape_before_bottleneck(input_shape).shape

        self.flatten_shape = torch.numel(self._calculate_shape_before_bottleneck(input_shape))

        self.flatten = nn.Flatten()

        self.mu = nn.Linear(self.flatten_shape, self.latent_space_dim)

        self.log_sigma = nn.Linear(self.flatten_shape, self.latent_space_dim)

        

    def _calculate_shape_before_bottleneck(self, input_shape: List[int]):

        x = torch.ones(input_shape)

        x = torch.unsqueeze(x, 0) 

        for convblock in self.convblocks:

            x = convblock(x)

        return x   

    def _reparameterized(self, mu, log_sigma):
        
        eps = torch.randn(size=mu.shape)
        
        sample_point = mu + torch.exp(log_sigma / 2) * eps
        
        return sample_point    

    def forward(self, x):

        for convblock in self.convblocks:
            
            x = convblock(x)

        x = self.flatten(x)

        mu = self.mu(x)

        log_sigma= self.log_sigma(x)

        x = self._reparameterized(mu, log_sigma)

        return x, (mu, log_sigma)




# DECODER

class Decoder(nn.Module):

    def __init__(self,
                latent_space_dim :int,
                shape_before_bottleneck : torch.Size,
                conv_filters : List[int], # must have 1 more element than others -> LAST element must be 3 for colored images
                conv_kernels : List[Tuple[int]],
                conv_strides : List[Tuple[int]],
                paddings : List[Tuple[int]],
                output_paddings : List[Tuple[int]],
                dilations : List[Tuple[int]],
                out_channel : int,
                **kwargs):

        super(Decoder, self).__init__(**kwargs)

        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.paddings = paddings
        self.output_paddings = output_paddings
        self.dilations = dilations
        self.out_channel = out_channel
        self.shape_before_bottleneck = shape_before_bottleneck

        self.flatten_shape = torch.numel(torch.ones(self.shape_before_bottleneck))
        self.fc = nn.Linear(latent_space_dim, self.flatten_shape)

        # dim assertion 
        assert len(self.conv_kernels) == len(self.conv_strides) == len(self.paddings) == len(self.output_paddings) == len(self.dilations)

        self.convtransposes = nn.Sequential(
            OrderedDict(
                [
            (
            f"Convolution_Transpose_Block{i+1}",          
            ConvTranspose2dBlock(conv_filters_in=self.conv_filters[i],
                                conv_filters_out=self.conv_filters[i+1],
                                conv_kernels=self.conv_kernels[i],
                                conv_strides=self.conv_strides[i],
                                paddings=self.paddings[i],
                                output_paddings=self.output_paddings[i],
                                dilations=self.dilations[i])
            )
            
            for i in range(len(self.conv_filters) - 1)

                ]
            )
        )

        self.output_convolution = nn.ConvTranspose2d(in_channels=self.conv_filters[-1],
                                                    out_channels=3, # colored images
                                                    kernel_size=self.conv_kernels[0],
                                                    stride=(1, 1),
                                                    padding=(1, 1),
                                                    output_padding=(0, 0),
                                                    dilation=(2, 2)

        )

    
    def forward(self, x):

        x = self.fc(x)
        
        x = x.view(self.shape_before_bottleneck)

        for convtransposeblock in self.convtransposes:

            x = convtransposeblock(x)

        x = self.output_convolution(x)

        x = torch.tanh(x)

        return x



# VAE

class VAE(nn.Module):

    def __init__(self,
                input_shape : List[int],
                conv_filters : List[Tuple[int]],
                conv_kernels : List[Tuple[int]],
                conv_strides : List[Tuple[int]],
                paddings : List[Tuple[int]],
                output_paddings : List[Tuple[int]],
                dilations: List[Tuple[int]],
                latent_space_dim : int,
                **kwargs):

        super(VAE, self).__init__(**kwargs)

        self.input_shape = input_shape
        
        self.latent_space_dim = latent_space_dim
        
        self.encoder = Encoder(input_shape=input_shape,
                                conv_filters=conv_filters,
                                conv_kernels=conv_kernels,
                                conv_strides=conv_strides,
                                paddings=paddings,
                                dilations=dilations,
                                latent_space_dim=latent_space_dim
                                )

        self.shape_before_bottleneck = self.encoder.shape_before_bottleneck

        self.decoder = Decoder(latent_space_dim=latent_space_dim,
                                shape_before_bottleneck=self.encoder.shape_before_bottleneck,
                                conv_filters=conv_filters[::-1],
                                conv_kernels=conv_kernels[::-1],
                                conv_strides=conv_strides[::-1],
                                paddings=paddings[::-1],
                                output_paddings=output_paddings,
                                dilations=dilations[::-1],
                                out_channel=3
                                )


    def forward(self, x):
        z, (mu, log_sigma) = self.encoder(x)
            
        x_prime = self.decoder(z)

        return z, mu, log_sigma, x_prime

    def sample(self, eps=None):

        if eps is None:
            eps = torch.randn([1, self.latent_space_dim])
            return self.decoder(eps)

        else:
            return self.decoder(eps)

    def reconstruct(self, images):
        latent_representations = self.encoder(images)
        reconstructed_images = self.decoder(latent_representations)

        return reconstructed_images, latent_representations

    @staticmethod
    def kl_div(mu, log_sigma):
        loss = -0.5 * torch.sum(1 + log_sigma + torch.square(mu) -torch.exp(log_sigma), 1)

        return loss
    
    def loss_fn(self, x, x_prime, mu, log_sigma):

        kld_loss = self.kl_div
        recon_loss = nn.MSELoss()

        kld = kld_loss(mu, log_sigma)
        recon = recon_loss(x, x_prime)

        loss = kld + recon

        return loss, kld, recon