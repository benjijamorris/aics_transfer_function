import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from  torch.nn.modules.upsampling import Upsample
import numpy as np
import pdb
import torch.nn.functional as F
import time

class UnetGenerator_upiso(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False,unet_upsampling=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator_upiso, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock_up(ngf * 4, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock_up(ngf * 4, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock_up(ngf * 4, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_up(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_up(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock_up(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer,unet_upsampling=unet_upsampling)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/upsampling.py
class Upsample3D(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample3D, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    # @weak_script_method
    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info

class UnetSkipConnectionBlock_up(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, unet_upsampling=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock_up, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=(3,3,3),
                             stride=(1,1,1), padding=(1,1,1), bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            if unet_upsampling:
                upconv = [nn.Conv3d(inner_nc * 2,inner_nc,
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=True),
                      Upsample3D(scale_factor=tuple([1.0,1.0,1.0]), mode='trilinear',align_corners=True),
                      norm_layer(inner_nc),
                      nn.ReLU(True),
                      nn.Conv3d(inner_nc, outer_nc, kernel_size=1, padding=0)]
            else:
                upconv = [nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                            kernel_size=(3,3,3), stride=(1,1,1),
                                            padding=(1,1,1)),]
            
                                            
            up = [uprelu,] + upconv + [nn.Tanh(),]
            down = [downconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=(3,3,3), stride=(1,1,1),
                                        padding=(1,1,1), bias=use_bias) #origin: 4,2,1
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=(3,3,3), stride=(1,1,1),
                                        padding=(1,1,1), bias=use_bias) #origin: 4,2,1
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

# ------------------------------
# *******************************
# ------------------------------


class UnetGenerator_downiso(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False,unet_upsampling=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator_downiso, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock_down(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock_down(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock_down(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_down(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_down(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock_down(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer,unet_upsampling=unet_upsampling)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock_down(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, unet_upsampling=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock_down, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=(1,1,1), bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            if unet_upsampling:
                upconv = [nn.Conv3d(inner_nc * 2,inner_nc,
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=True),
                      Upsample3D(scale_factor=tuple([2.0,2.0,2.0]), mode='trilinear',align_corners=True),
                      norm_layer(inner_nc),
                      nn.ReLU(True),
                      nn.Conv3d(inner_nc, outer_nc, kernel_size=1, padding=0)]
            else:
                upconv = [nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=(1,1,1)),]
            
                                            
            up = [uprelu,] + upconv + [nn.Tanh(),]
            down = [downconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=(1,1,1), bias=use_bias) #origin: 4,2,1
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=(1,1,1), bias=use_bias) #origin: 4,2,1
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
