import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.modules.upsampling import Upsample
import torch.nn.functional as F
import math
import numbers


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance

    For BatchNorm, we use learnable affine parameters and track running statistics
    (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track
    running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm3d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm3d, affine=False, track_running_stats=False
        )
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass
                              of BaseOptions．lr_policy is the name of learning rate
                              policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <niter> epochs
    and linearly decay the rate to zero over the next <niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch
    schedulers. See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.training_setting["lr_policy"] == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.training_setting["niter"]) / float(
                opt.training_setting["niter_decay"] + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.training_setting["lr_policy"] == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.training_setting["lr_decay_iters"], gamma=0.1
        )
    elif opt.training_setting["lr_policy"] == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opt.training_setting["lr_policy"] == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.training_setting["niter"], eta_min=0
        )
    else:
        return NotImplementedError("your learning rate policy is not implemented")
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method:
                           normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming
    might work better for some applications. Feel free to try yourself.
    """
    # define the initialization function
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("init method is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm3d") != -1:
            # Note: BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method:
                              normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    assert torch.cuda.is_available(), "GPU is not available"
    net.to(torch.device("cuda:0"))
    # net = torch.nn.DataParallel(net, torch.device('cuda:0'))  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


# TODO: add type hint
def define_G(
    input_nc,
    output_nc,
    ngf,
    netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    unet_upsampling=True,
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name:
                      resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network:
                      batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256]
        (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks)
                                and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between
        a few downsampling/upsampling operations. We adapt Torch code from
        Justin Johnson's neural style transfer project
        (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks":
        net = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
        )
    elif netG == "r9cr":
        net = ResnetGeneratorCR(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
        )
    elif netG == "resnopad":
        net = ResnetGeneratorNopad(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
        )
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=6,
        )
    elif netG == "unet_64":
        net = UnetGenerator(
            input_nc,
            output_nc,
            6,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            unet_upsampling=unet_upsampling,
        )
    elif netG == "unet_128":
        net = UnetGenerator(
            input_nc,
            output_nc,
            7,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            unet_upsampling=unet_upsampling,
        )
    elif netG == "unet_256":
        net = UnetGenerator(
            input_nc,
            output_nc,
            8,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            unet_upsampling=unet_upsampling,
        )
    elif netG == "unet_512_nopad":  # (32,512,512) --> (28,152,152)
        net = UnetGeneratorNopad(
            input_nc,
            output_nc,
            7,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            unet_upsampling=unet_upsampling,
        )
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return init_net(net, init_type, init_gain)


# TODO: add type hint
def define_NG(
    input_nc,
    output_shape,
    ngf,
    netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
):
    net = NoiseResnetGenerator(1, output_shape)
    return init_net(net, init_type, init_gain)


# TODO: add type hint
def define_S(
    input_nc,
    ndf,
    n_layers_D=3,
    norm="batch",
    init_type="normal",
    init_gain=0.02,
    device=None,
):
    norm_layer = get_norm_layer(norm_type=norm)
    net = FindOffsets(input_nc, ndf, n_layers_D, norm_layer=norm_layer, device=device)
    return init_net(net, init_type, init_gain)


# def shift(tensor,offsets_zyx,device=None):
#     '''
#     dz>0: stack goes up (-z:-1)=padding
#     dy>0: goes up; dx>0: goes right
#     '''
#     assert len(tensor.shape) == 5
#     dz,dy,dx = offsets_zyx.type(torch.float32)
#     nz,ny,nx = tensor.shape[2:]
#     z_linspace = torch.linspace(-1,1,steps=nz,device=device)+dz*2.0/nz
#     y_linspace = torch.linspace(-1,1,steps=ny,device=device)+dy*2.0/ny
#     x_linspace = torch.linspace(-1,1,steps=nx,device=device)+dx*2.0/nx
#     z_grid, y_grid, x_grid = torch.meshgrid(z_linspace, y_linspace, x_linspace)
#     grid = torch.cat([x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), \
#            z_grid.unsqueeze(-1)], dim=-1).unsqueeze(0)
#     return F.grid_sample(tensor, grid,padding_mode='reflection')


class FindOffsets(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, device=None
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(FindOffsets, self).__init__()
        self.device = device
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 0
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 3, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)
        # self.Linear_m1 = MLP( ndf * nf_mult//2, 3, mid_dim=16, n_blk=3)
        self.base_grid = self.build_base_meshgrid()
        self.grid = torch.zeros_like(self.base_grid)

    def build_base_meshgrid(self):
        nz = 32
        ny = 256
        nx = 256
        z_linspace = torch.linspace(-1, 1, steps=nz, device=self.device)
        y_linspace = torch.linspace(-1, 1, steps=ny, device=self.device)
        x_linspace = torch.linspace(-1, 1, steps=nx, device=self.device)
        z_grid, y_grid, x_grid = torch.meshgrid(z_linspace, y_linspace, x_linspace)
        grid = torch.cat(
            [x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), z_grid.unsqueeze(-1)], dim=-1
        ).unsqueeze(0)
        return grid

    def shift(self, tensor, offsets_zyx):
        """
        dz>0: stack goes up (-z:-1)=padding
        dy>0: goes up; dx>0: goes right
        """
        assert len(tensor.shape) == 5
        dz, dy, dx = offsets_zyx.type(torch.float32)
        nz, ny, nx = tensor.shape[2:]
        z_linspace = torch.linspace(-1, 1, steps=nz, device=self.device) + dz * 2.0 / nz
        y_linspace = torch.linspace(-1, 1, steps=ny, device=self.device) + dy * 2.0 / ny
        x_linspace = torch.linspace(-1, 1, steps=nx, device=self.device) + dx * 2.0 / nx
        z_grid, y_grid, x_grid = torch.meshgrid(z_linspace, y_linspace, x_linspace)
        grid = torch.cat(
            [x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), z_grid.unsqueeze(-1)], dim=-1
        ).unsqueeze(0)
        return F.grid_sample(tensor, grid, padding_mode="border")

    def forward(self, input, base):
        """Standard forward."""
        assert input.shape[0] == 1, "only support batch_size=1 for stn"
        pred = self.model(input)
        self.offsets = torch.mean(pred, dim=(2, 3, 4), keepdim=False, out=None)[0]
        aligned = self.shift(base, self.offsets)
        # offsets = self.Linear_m1(offsets)
        return aligned, self.offsets

    def adjust(self, base, offsets):
        aligned = self.shift(base, offsets)
        return aligned

    def get_offsets(self, input):
        assert input.shape[0] == 1, "only support batch_size=1 for stn"
        pred = self.model(input)
        offsets = torch.mean(pred, dim=(2, 3, 4), keepdim=False, out=None)[0]
        return offsets


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        print(f"GaussianSmoothing: {kernel.shape}")
        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class NoiseResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_shape,
        ngf=64,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        super(NoiseResnetGenerator, self).__init__()
        sz, sy, sx = output_shape
        # self.Nb1 = NoiseBlock(16, -1, 32,(4,4,4),16)
        self.Nb1 = NoiseBlock(16, -1, 32, (sz // 8, sy // 32, sx // 32), 16)
        # Nb1 (B, 16, 4, 4, 4)
        Trans1 = [
            nn.ConvTranspose3d(
                16, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ),
            norm_layer(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(
                16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ),
            norm_layer(8),
            nn.ReLU(True),
        ]
        self.Trans1 = nn.Sequential(*Trans1)
        # Trans1, (B, 8, 16,16,16),
        # self.Nb2 = NoiseBlock(32,8,32,(16,16,16),8)
        self.Nb2 = NoiseBlock(32, 8, 32, (sz // 2, sy // 8, sx // 8), 8)
        Trans2_1 = [
            nn.ConvTranspose3d(
                8, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ),
            norm_layer(8),
            nn.ReLU(True),
        ]
        self.Trans2_1 = nn.Sequential(*Trans2_1)
        Trans2_2 = [
            nn.ConvTranspose3d(
                8,
                8,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
                bias=True,
            ),
            norm_layer(8),
            nn.ReLU(True),
        ]
        self.Trans2_2 = nn.Sequential(*Trans2_2)
        # self.Nb3 = NoiseBlock(48,8,32,(32,64,64),4)
        self.Nb3 = NoiseBlock(48, 8, 32, (sz, sy // 2, sx // 2), 4)
        Trans3 = [
            nn.ConvTranspose3d(
                4,
                4,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
                bias=True,
            ),
            norm_layer(4),
            nn.ReLU(True),
        ]
        self.Trans3 = nn.Sequential(*Trans3)
        self.Nb4 = NoiseBlock(48, 4, 32, (sz, sy, sx), 1)
        Trans4 = [
            nn.Tanh(),
        ]
        self.Trans4 = nn.Sequential(*Trans4)

    def forward(self, z, mask):
        """Standard forward"""
        nb1_1_tensor = self.Nb1(None, z[:, :16], mask)
        nb1_2_tensor = self.Trans1(nb1_1_tensor)

        nb2_1_tensor = self.Nb2(nb1_2_tensor, z[:, 16:48], mask)
        nb2_2_tensor = self.Trans2_1(nb2_1_tensor)
        nb2_3_tensor = self.Trans2_2(nb2_2_tensor)

        nb3_1_tensor = self.Nb3(nb2_3_tensor, z[:, 48:96], mask)
        nb3_2_tensor = self.Trans3(nb3_1_tensor)

        nb4_1_tensor = self.Nb4(nb3_2_tensor, z[:, 96:144], mask)
        nb4_2_tensor = self.Trans4(nb4_1_tensor)
        return nb4_2_tensor


class NoiseBlock(nn.Module):
    def __init__(
        self,
        input_noise_nc,
        input_nc,
        ngf,
        output_shape,
        output_nc,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        super(NoiseBlock, self).__init__()
        self.output_shape = output_shape
        self.output_nc = output_nc
        dim1 = output_shape[0] * output_shape[1] * output_shape[2] * output_nc
        self.Linear_m = MLP(input_noise_nc, dim1, mid_dim=16, n_blk=3)

        Res_m1 = [
            nn.Conv3d(
                output_nc + 1, ngf, kernel_size=3, stride=1, padding=1, bias=True
            ),
            norm_layer(ngf),
            nn.ReLU(True),
            ResnetBlock(
                ngf,
                padding_type=padding_type,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                use_bias=True,
            ),
        ]
        self.Res_m1 = nn.Sequential(*Res_m1)

        if input_nc > 0:
            Res_m2 = [
                nn.Conv3d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ngf),
                nn.ReLU(True),
                ResnetBlock(
                    ngf,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=True,
                ),
            ]
            self.Res_m2 = nn.Sequential(*Res_m2)
        else:
            self.Res_m2 = None

        Res_m3 = [
            ResnetBlock(
                ngf,
                padding_type=padding_type,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                use_bias=True,
            ),
            nn.Conv3d(ngf, output_nc, kernel_size=1, stride=1, padding=0, bias=True),
        ]
        self.Res_m3 = nn.Sequential(*Res_m3)

    def forward(self, input, z, mask):
        z2 = self.Linear_m(z)
        z2 = z2.view(
            (
                -1,
                self.output_nc,
                self.output_shape[0],
                self.output_shape[1],
                self.output_shape[2],
            )
        )
        mask2 = F.interpolate(
            mask, size=self.output_shape, mode="trilinear", align_corners=True
        )
        cat2 = torch.cat((z2, mask2), dim=1)
        if torch.is_tensor(input):
            res = self.Res_m1(cat2) + self.Res_m2(input)
        else:
            res = self.Res_m1(cat2)
        res2 = self.Res_m3(res)
        return res2


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm="none", activation="relu"):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(norm_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(
        self, input_dim, output_dim, mid_dim, n_blk, norm="none", activ="relu"
    ):

        super(MLP, self).__init__()
        assert n_blk >= 2
        self.model = []
        self.model += [LinearBlock(input_dim, mid_dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(mid_dim, mid_dim, norm=norm, activation=activ)]
        self.model += [
            LinearBlock(mid_dim, output_dim, norm="none", activation="none")
        ]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def define_D(
    input_nc, ndf, netD, n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02
):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator;
                              effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in
        the discriminator with the parameter <n_layers_D> (default=3 as used in [basic]
        (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for
    non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=1, norm_layer=norm_layer)
    elif netD == "basic433":  # default PatchGAN classifier
        net = NLayerDiscriminator433(input_nc, ndf, n_layers=1, norm_layer=norm_layer)
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == "pixel":  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError("your discriminator model is not recognized")
    return init_net(net, init_type, init_gain)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla,
                                lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or
                                      fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or
                                      fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(
    netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0
):
    """Calculate the gradient penalty loss, used in WGAN-GP paper
        https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU
        type (str)                  -- if we mix real and fake data or not
                                       [real | fake | mixed].
        constant (float)            -- the constant used in formula
                                       ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = (
                alpha.expand(
                    real_data.shape[0], real_data.nelement() // real_data.shape[0]
                )
                .contiguous()
                .view(*real_data.shape)
            )
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError("{} not implemented".format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolatesv,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (
            ((gradients + 1e-16).norm(2, dim=1) - constant) ** 2
        ).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few
       downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer
    project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers:
                                    reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # TODO: switch to ReflectionPad3d
        model = [
            nn.ReplicationPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # self.debug = nn.Sequential(*model)

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv3d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose3d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReplicationPad3d(3)]
        # TODO: switch to ReflectionPad3d
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        # de = self.debug(input)
        return self.model(input)


class ResnetGeneratorCR(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a
       few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer
    project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers:
                                   reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGeneratorCR, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # TODO: switch to ReflectionPad3d
        model = [
            nn.ReplicationPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # self.debug = nn.Sequential(*model)

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv3d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=(2, 2, 2),
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling - 1):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose3d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        mult = 2
        model += [
            nn.Conv3d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            ),
            Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]
        model += [nn.ReplicationPad3d(3)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        # de = self.debug(input)
        return self.model(input)


class ResnetGeneratorNopad(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few
       downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer
    project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        n_blocks=6,
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers:
                                    reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGeneratorNopad, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # TODO: switch to ReflectionPad3d
        model = [
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # self.debug = nn.Sequential(*model)

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv3d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=(1, 2, 2),
                    padding=0,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type="zero",
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.Conv3d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=use_bias,
                ),
                Upsample3D(
                    scale_factor=tuple([1.0, 2.0, 2.0]),
                    mode="trilinear",
                    align_corners=True,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=(3, 7, 7), padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        # de = self.debug(input)
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )
        self.padding_type = padding_type

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer,
        and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReplicationPad3d(1)]
            # TODO: switch to ReflectionPad3d
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == "zero":
            p = 1
        elif padding_type == "none":
            p = 0
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReplicationPad3d(1)]
            # TODO: switch to ReflectionPad3d
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == "zero":
            p = 1
        elif padding_type == "none":
            p = 0
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        # print('x',x.shape)
        if self.padding_type == "none":
            print(x.shape)
            print(self.conv_block(x).shape)
            out = x[:, :, 2:-2, 2:-2, 2:-2] + self.conv_block(x)
        else:
            out = x + self.conv_block(x)  # add skip connections
        # print('out',out.shape)
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        unet_upsampling=True,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example,
                                # if |num_downs| == 7, image of size 128x128 will
                                # become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            unet_upsampling=unet_upsampling,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetGeneratorNopad(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        unet_upsampling=True,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example,
                                # if |num_downs| == 7, image of size 128x128 will
                                # become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        # Unet128 nopad: (32,512,512) --> (28,152,152)
        super(UnetGeneratorNopad, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlockNopad(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlockNopad(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                downsampling=False,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlockNopad(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlockNopad(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlockNopad(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlockNopad(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            unet_upsampling=unet_upsampling,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/upsampling.py
class Upsample3D(nn.Module):
    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
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
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners
        )

    def extra_repr(self):
        if self.scale_factor is not None:
            info = "scale_factor=" + str(self.scale_factor)
        else:
            info = "size=" + str(self.size)
        info += ", mode=" + self.mode
        return info


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        unet_upsampling=True,
    ):
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
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(
            input_nc,
            inner_nc,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            bias=use_bias,
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            if unet_upsampling:
                upconv = [
                    nn.Conv3d(
                        inner_nc * 2,
                        inner_nc,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    Upsample3D(
                        scale_factor=tuple([1.0, 2.0, 2.0]),
                        mode="trilinear",
                        align_corners=True,
                    ),
                    norm_layer(inner_nc),
                    nn.ReLU(True),
                    nn.Conv3d(inner_nc, outer_nc, kernel_size=1, padding=0),
                ]
            else:
                upconv = [
                    nn.ConvTranspose3d(
                        inner_nc * 2,
                        outer_nc,
                        kernel_size=(3, 4, 4),
                        stride=(1, 2, 2),
                        padding=(1, 1, 1),
                    )
                ]

            up = [uprelu] + upconv + [nn.Tanh()]
            down = [downconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(
                inner_nc,
                outer_nc,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=use_bias,
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(
                inner_nc * 2,
                outer_nc,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=use_bias,
            )
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
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlockNopad(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
        unet_upsampling=True,
        downsampling=True,
    ):
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
        super(UnetSkipConnectionBlockNopad, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(
            input_nc,
            inner_nc,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=0,
            bias=use_bias,
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            if unet_upsampling:
                upconv = [
                    nn.Conv3d(
                        inner_nc * 2,
                        inner_nc,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                    Upsample3D(
                        scale_factor=tuple([1.0, 2.0, 2.0]),
                        mode="trilinear",
                        align_corners=True,
                    ),
                    norm_layer(inner_nc),
                    nn.ReLU(True),
                    nn.Conv3d(inner_nc, outer_nc, kernel_size=1, padding=0),
                ]
            else:
                upconv = [
                    nn.ConvTranspose3d(
                        inner_nc * 2,
                        outer_nc,
                        kernel_size=(3, 4, 4),
                        stride=(1, 2, 2),
                        padding=0,
                    )
                ]

            up = [uprelu] + upconv + [nn.Tanh()]
            down = [downconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(
                inner_nc,
                outer_nc,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=0,
                bias=use_bias,
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        elif downsampling:
            upconv = nn.ConvTranspose3d(
                inner_nc * 2,
                outer_nc,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=0,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        else:
            upconv = nn.ConvTranspose3d(
                inner_nc * 2,
                outer_nc,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=0,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            out = self.model(x)
            in_z, in_y, in_x = x.shape[2:]
            out_z, out_y, out_x = out.shape[2:]
            z1 = (in_z - out_z) // 2
            z2 = z1 + out_z
            y1 = (in_y - out_y) // 2
            y2 = y1 + out_y
            x1 = (in_x - out_x) // 2
            x2 = x1 + out_x
            return torch.cat([x[:, :, z1:z2, y1:y2, x1:x2], out], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerDiscriminator433(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator433, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = (4, 3, 3)
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.InstanceNorm3d
        else:
            use_bias = norm_layer != nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
