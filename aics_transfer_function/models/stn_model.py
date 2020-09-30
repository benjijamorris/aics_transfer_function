import torch
from .base_model import BaseModel
from . import networks
from skimage.filters import threshold_otsu
import numpy as np
import pdb


class StnModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.offsetlogpath = opt.offsetlogpath
        self.netG_name = opt.netG
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'S_L1', 'S_reg', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D','S']
        else:  # during test time, only load G
            self.model_names = ['G','S']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netS = networks.define_S(opt.input_nc*2 , opt.ndf, 3, opt.norm, opt.init_type, opt.init_gain,self.device, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1_aligned = torch.nn.L1Loss()
            self.criterionL1_offsets = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_GS = torch.optim.Adam(list(self.netG.parameters()) + list(self.netS.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_S)
            self.optimizers.append(self.optimizer_GS)
            
            # stage1: train p2p only, ignore stn; loss(self.fake_B1,self.real_B)
            # stage2: freeze p2p, train stn only; fake_B2 = stn(self.fake_B1.detach()), loss(self.fake_B2,self.real_B)
            # stage3: train p2p (fine tune), freeze stn; loss(self.fake_B2,self.real_B)
            self.stage = 1

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.calcAveOffset = input['calcAveOffset']
        self.fnnA = input['A_short_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B0 = self.netG(self.real_A)  # G(A)
        # realfake_B = torch.cat((self.fake_B0,self.real_B),1)
        # # self.offsets = self.netS(realfake_B)
        # # self.fake_B = networks.shift(self.fake_B0,self.offsets, self.device)
        # realfake_B = torch.cat((self.fake_B0.detach(),self.real_B),1)
        # # self.offsets = self.netS(realfake_B)
        # # self.fake_B = networks.shift(self.fake_B0,self.offsets,self.device)
        # self.fake_B = self.netS(realfake_B,self.fake_B0.detach())
        pass


    def crop_edge(self,img1,img2):
        #crop img1 in the center, s.t. img1.shape=img2.shape
        in_z,in_y,in_x = img1.shape[2:]
        out_z,out_y,out_x = img2.shape[2:]
        z1 = (in_z-out_z)//2
        z2 = z1+out_z
        y1 = (in_y-out_y)//2
        y2 = y1+out_y
        x1 = (in_x-out_x)//2
        x2 = x1+out_x
        return img1[:,:,z1:z2,y1:y2,x1:x2]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if 'nopad' in self.netG_name:
            self.real_A = self.crop_edge(self.real_A,self.fake_B0)
            self.real_B = self.crop_edge(self.real_B,self.fake_B0)
        fake_AB = torch.cat((self.real_A, self.fake_B0), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
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
        fake_AB = torch.cat((self.real_A, self.fake_B0), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B0, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def backward_GS(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B0), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_S_L1 = self.criterionL1_aligned(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_GS = self.loss_G_GAN + self.loss_S_L1
        self.loss_GS.backward()

    def backward_S(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_S_L1 = self.criterionL1_aligned(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_S_reg = self.criterionL1_offsets(self.offsets,torch.zeros_like(self.offsets,dtype=self.offsets.dtype))* self.opt.stn_lambda_reg
        # combine loss and calculate gradients
        self.loss_S =  self.loss_S_L1 +self.loss_S_reg 
        self.loss_S.backward()

    def optimize_parameters(self):
        if self.stage == 1:
            self.optimize_parameters_s1()
        elif self.stage == 2:
            self.optimize_parameters_s2()
        elif self.stage == 3:
            self.optimize_parameters_s3()
        else:
            raise NotImplementedError("stage should be 1,2,3")

    def optimize_parameters_s1(self):
        print('stage1')
        self.set_requires_grad(self.netS, False) 
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netG, True) 
        self.fake_B0 = self.netG(self.real_A)  # G(A)
        self.fake_B = self.fake_B0

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netG, False) 
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True) 
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def optimize_parameters_s2(self):
        print('stage2, pre-train Alignment module')
        self.set_requires_grad(self.netG, False)  # enable backprop for D
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netS, True) 

        self.fake_B0 = self.netG(self.real_A)  # G(A)
        realfake_B = torch.cat((self.fake_B0.detach(),self.real_B),1)
        self.fake_B,self.offsets = self.netS(realfake_B,self.fake_B0.detach())
        print(self.offsets)

        # if self.calcAveOffset and ratio >= 0.01:
        if self.calcAveOffset:
            z,y,x = self.offsets.detach().cpu().numpy()
            with open(self.offsetlogpath, 'a') as fp:
                fp.write(f'{self.fnnA},{z},{y},{x}\n')

        self.optimizer_S.zero_grad()        # set G's gradients to zero
        self.backward_S()                   # calculate graidents for G
        self.optimizer_S.step()             # udpate G's weights

    def optimize_parameters_s3(self):
        print('stage3')
        self.fake_B0 = self.netG(self.real_A)  # G(A)
        realfake_B = torch.cat((self.fake_B0,self.real_B),1)
        self.fake_B,self.offsets = self.netS(realfake_B,self.fake_B0)
        print(self.offsets)

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netG, False) 
        self.set_requires_grad(self.netS, False) 
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True) 
        if self.opt.stn_fix_stn_model:
            self.set_requires_grad(self.netS, False) 
        else:
            self.set_requires_grad(self.netS, True) 
        self.optimizer_GS.zero_grad()        # set G's gradients to zero
        self.backward_GS()                   # calculate graidents for G
        self.optimizer_GS.step()             # udpate G's weights

        #just a placeholder for output. Have no impact on anything
        # self.loss_S_reg = 0.0
        # self.loss_S_L1 = 0.0
    
    def get_shift(self):
        return self.offsets

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.fake_B0 = self.netG(self.real_A)  # G(A)
            realfake_B = torch.cat((self.fake_B0,self.real_B),1)
            self.fake_B,self.offsets = self.netS(realfake_B,self.fake_B0)
            return (self.real_A, self.real_B, self.fake_B0, self.fake_B)

    # def shift(self,tensor,offsets_zyx):
    #     '''
    #     dz>0: stack goes up (-z:-1)=padding
    #     dy>0: goes up; dx>0: goes right
    #     '''
    #     assert len(tensor.shape) == 5
    #     dz,dy,dx = offsets_zyx.type(torch.float64)
    #     nz,ny,nx = tensor.shape[2:]
    #     z_linspace = torch.linspace(-1,1,steps=nz,device=self.device)+dz*2.0/nz     # linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) 
    #     y_linspace = torch.linspace(-1,1,steps=ny,device=self.device)+dy*2.0/ny
    #     x_linspace = torch.linspace(-1,1,steps=nx,device=self.device)+dx*2.0/nx
    #     z_grid, y_grid, x_grid = torch.meshgrid(z_linspace, y_linspace, x_linspace)
    #     grid = torch.cat([x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), z_grid.unsqueeze(-1)], dim=-1).unsqueeze(0)
    #     return F.grid_sample(tensor, grid,padding_mode='reflection')
    #     # original_tensor = from_numpy(original).unsqueeze(0).unsqueeze(0)
    #     # offsets = from_numpy(np.array([0.5,0,0]))
    #     # new_tensor = shift(original_tensor,offsets)