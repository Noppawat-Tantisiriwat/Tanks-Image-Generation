from torch import nn
import torch.nn.functional as F
from typing import Tuple




# convolution block -> repeated unit infrastructure of ENCODER

class Conv2dBlock(nn.Module):   # conv -> maxpool -> batchnorm

    def __init__(self,
                conv_filters_in: int, # -> sets of input filters
                conv_filters_out: int, # -> sets of output filters
                conv_kernels: Tuple[int], # -> sets of convolution kernels
                conv_strides: Tuple[int], # -> sets of convolution strides
                paddings: Tuple[int], # -> sets of paddings
                dilations: Tuple[int], # sets of dilations
                **kwargs):
        
        super(Conv2dBlock, self).__init__(**kwargs) # super initiate

        self.conv_filters_in = conv_filters_in 
        self.conv_filters_out = conv_filters_out
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.paddings = paddings
        self.dilations = dilations

        # parsing all artributes to PyTorch Modules
        self.conv2d = nn.Conv2d(in_channels=self.conv_filters_in, 
                                out_channels=self.conv_filters_out, 
                                kernel_size=self.conv_kernels,
                                stride=self.conv_strides,
                                padding=self.paddings,
                                dilation=self.dilations)

        self.maxpool2d = nn.MaxPool2d(kernel_size=self.conv_kernels, 
                                    stride=self.conv_strides,
                                    padding=self.paddings,
                                    dilation=self.dilations)

        self.batchnorm = nn.BatchNorm2d(num_features=self.conv_filters_out)

    # def forward passing of the unit
    def forward(self, x):

        x = self.conv2d(x)

        x = self.maxpool2d(x)

        x = self.batchnorm(x)

        x = F.relu(x)

        return x





# convolution transpose block -> repeated unit infrastructure of DECODER

class ConvTranspose2dBlock(nn.Module):

    def __init__(self, 
                conv_filters_in: int, # -> sets of convolution input filters
                conv_filters_out: int, # -> sets of convolution output filters
                conv_kernels: Tuple[int], # -> sets of convolution kernels
                conv_strides: Tuple[int], # sets of convolution strides
                paddings: Tuple[int], # sets of paddings
                output_paddings: Tuple[int], # sets of padding after convolution
                dilations: Tuple[int], # sets of dilation
                **kwargs):

        super(ConvTranspose2dBlock, self).__init__(**kwargs) # super initiate

        self.conv_filters_in = conv_filters_in
        self.conv_filters_out = conv_filters_out
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.paddings = paddings
        self.output_paddings = output_paddings
        self.dilations = dilations

        # parsing all artributes to PyTorch Modules
        self.convtranspose2d = nn.ConvTranspose2d(in_channels=self.conv_filters_in, 
                                                out_channels=self.conv_filters_out, 
                                                kernel_size=self.conv_kernels,
                                                stride=self.conv_strides,
                                                padding=self.paddings,
                                                output_padding=self.output_paddings,
                                                dilation=self.dilations)

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=self.conv_strides[0])


        self.batchnorm = nn.BatchNorm2d(num_features=self.conv_filters_out)

    # define forward passing
    def forward(self, x):

        x = self.convtranspose2d(x)

        x = self.upsampling(x)

        x = self.batchnorm(x)

        x = F.relu(x)

        return x



