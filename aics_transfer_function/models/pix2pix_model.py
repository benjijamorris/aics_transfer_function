import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """
    This class implements the pix2pix model, for learning a mapping from
    input images to output images given paired data. pix2pix paper:
    https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be
            a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the models you want to save to the disk. The training/test scripts
        # will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
            train_opt = opt.training_setting
        else:  # during test time, only load G
            self.model_names = ["G"]

        net_opt = opt.network
        self.net_opt = net_opt
        self.netG_name = net_opt["netG"]
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        # specify the images you want to save/display. The training/test scripts
        # will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B"]

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            net_opt["input_nc"],
            net_opt["output_nc"],
            net_opt["ngf"],
            net_opt["netG"],
            net_opt["norm"],
            not net_opt["no_dropout"],
            net_opt["init_type"],
            net_opt["init_gain"],
        )

        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input
            # and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(
                net_opt["input_nc"] + net_opt["output_nc"],
                net_opt["ndf"],
                net_opt["netD"],
                net_opt["n_layers_D"],
                net_opt["norm"],
                net_opt["init_type"],
                net_opt["init_gain"],
            )

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(net_opt["gan_mode"]).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created
            # by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=train_opt["lr"],
                betas=(train_opt["beta1"], 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=train_opt["lr"],
                betas=(train_opt["beta1"], 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = True  # self.opt.direction == 'AtoB'
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """
        Run forward pass
        called by both functions <optimize_parameters> and <test>.
        """
        self.fake_B = self.netG(self.real_A)  # G(A)

    def crop_edge(self, img1, img2):
        in_z, in_y, in_x = img1.shape[2:]
        out_z, out_y, out_x = img2.shape[2:]
        z1 = (in_z - out_z) // 2
        z2 = z1 + out_z
        y1 = (in_y - out_y) // 2
        y2 = y1 + out_y
        x1 = (in_x - out_x) // 2
        x2 = x1 + out_x
        return img1[:, :, z1:z2, y1:y2, x1:x2]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if "nopad" in self.netG_name:
            self.real_A = self.crop_edge(self.real_A, self.fake_B)
            self.real_B = self.crop_edge(self.real_B, self.fake_B)
        # we use conditional GANs; we need to feed both input
        # and output to the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = (
            self.criterionL1(self.fake_B, self.real_B) * self.net_opt["lambda_L1"]
        )
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)
        # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save
        intermediate steps for backprop. It also calls <compute_visuals> to
        produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            return (self.real_A, self.real_B, self.fake_B)
