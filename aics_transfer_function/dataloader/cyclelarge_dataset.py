import numpy as np
import os
import os.path
from tifffile import imread, imsave
import random

import torch
from torch import from_numpy
from  torch.nn.modules.upsampling import Upsample
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from aicsimageio import AICSImage
from skimage.filters import threshold_otsu

import time

from scipy.stats import norm
from collections import OrderedDict
from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter


class cyclelargeDataset(Dataset):
    '''
    Dataloader for reading the training data and returns dataset class
    '''
    def __init__(self, opt, aligned=False): 
        Dataset.__init__(self)
        self.imgA = []
        self.imgB = []
        self.size_in = opt.size_in
        self.opt = opt
        self.filenamesA = None
        self.filenamesB = None
        self.name = opt.name
        self.batch_size = opt.batch_size
        self.aligned = aligned
        self.resizeA = opt.resizeA
        self.netG = opt.netG
        self.model = opt.model
        self.sample_mode = opt.sample_mode
        self.shift_dict = {}
        self.stn_adjust_dict = {} #similar to shift_dict, only for stn opt.stn_adjust_image use
        self.datarange_11 =  opt.datarange_11

        if self.model == 'pix2pix' and self.netG=='unet_512_nopad':
            self.up_scale = (1,1,1)
            self.size_out = (self.size_in[0]-4,152,152) #size_in = (32,512,512) --> (28,152,152)
        else:
            self.up_scale = (1,1,1)
            self.size_out = self.size_in

        if opt.model in ['pix2pix','stn']:
            self.aligned = True

        self.ToApplyShift = False #for opt.sample_mode == 'shift_odd'

    def simple_norm(self, struct_img, scaling_param, inplace=True):
        if not inplace:
            struct_img = np.copy(struct_img)
        print(f'intensity normalization: normalize into [mean - {scaling_param[0]} x std, mean + {scaling_param[1]} x std] ')
        
        ##########################################################
        # take middle chunk
        struct_img = np.squeeze(struct_img)
        img_smooth = gaussian_filter(struct_img.astype(np.float32), sigma=1.0, mode='nearest', truncate=3.0)
        th = threshold_otsu(img_smooth)
        img_bw = img_smooth > th
        low_chunk = 0
        high_chunk = struct_img.shape[0]
        for zz in range(struct_img.shape[0]):
            if np.count_nonzero(img_bw[zz,:,:]>0) > 50:
                if zz>0:
                    low_chunk = zz - 1
                break
        
        for zz in range(struct_img.shape[0]):
            if np.count_nonzero(img_bw[struct_img.shape[0]-zz-1,:,:]>0) > 50:
                if zz>0:
                    high_chunk = struct_img.shape[0] - zz 
                break

        structure_img0 = struct_img[low_chunk:high_chunk, :, :]
        ##########################################################

        m = np.mean(structure_img0)
        s = np.std(structure_img0)

        lower = max(m - scaling_param[0] * s, struct_img.min())
        upper = min(m + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img<lower] = lower
        struct_img[struct_img>upper] = upper
        struct_img = (struct_img - lower + 1e-8) / (upper - lower + 1e-8)
        if self.datarange_11:
            struct_img = struct_img * 2.0 - 1.0
        return np.expand_dims(struct_img, axis=0).astype('float32')


    def load_from_file(self, filenamesA, filenamesB, num_patch):
        norm_passed = [float(x) for x in self.opt.norm_factor.split(" ")]
        rint = lambda x: int(np.round(x))
        print('load %d patches'%num_patch)
        self.imgA = []
        self.imgB = []
        self.imgA_path = []
        self.imgA_short_path =[]
        self.imgB_path = []

        if self.name != 'denoise':
            assert len(filenamesA) == len(filenamesB)
        else:
            fdict = self.get_fdict(filenamesA, filenamesB)
        num_data = len(filenamesA)
        assert num_data>0
        self.num_patch_per_img = np.zeros((num_data,), dtype=int)

        if num_patch == -1:
            Stitch = True
        else:
            Stitch = False
            if num_data >= num_patch:
                print('suggest to use more patch in each buffer')
                self.num_patch_per_img[:num_patch]=1
            else:
                basic_num = num_patch // num_data
                self.num_patch_per_img[:] = basic_num
                self.num_patch_per_img[:(num_patch-basic_num*num_data)] = self.num_patch_per_img[:(num_patch-basic_num*num_data)] + 1

        if (not self.aligned) and (not Stitch):
            shuffle(filenamesA)
            shuffle(filenamesB)
        self.filenamesA = filenamesA
        self.filenamesB = filenamesB
        self.for_calc_ave_offset = [] # True if this patch is to be used for calculating mean offset for AutoAlign

        if self.opt.stn_adjust_fixed_z:
            print(f'read offsets from {self.opt.readoffsetfrom}')
            assert os.path.isfile(self.opt.readoffsetfrom) , f'opt.readoffsetfrom path: \'{self.opt.readoffsetfrom}\' is not found! \
                If you want to align images, set the correct path. If not, set opt.stn_adjust_fixed_z=False'
            with open(self.opt.readoffsetfrom,'r') as fp:
                fixed_dict1 = {}
                for line in fp.readlines():
                    key,z,y,x = line.strip().split(',')
                    z = float(z); y = float(y); x = float(x)
                    if key in fixed_dict1:
                        fixed_dict1[key].append([z,y,x])
                    else:
                        fixed_dict1[key] = [[z,y,x],]
                for key in fixed_dict1:
                    clip_param = np.percentile(np.array(fixed_dict1[key])[:,0], [10,90]) # only use the parameters that are between [10,90] percents.
                    offset_raw = []
                    for i in range(len(fixed_dict1[key])):
                        if fixed_dict1[key][i][0] > clip_param[0] and fixed_dict1[key][i][0] < clip_param[1]:
                            offset_raw.append(fixed_dict1[key][i])
                    offset_raw = np.array(offset_raw)
                    z_std = np.std(offset_raw, axis = 0)[0]

                    if z_std > 0.5:
                        print(f"WARNING: The standard deviation of offsets estimation for {key} is {z_std}. Not accurate!")
                        with open(self.opt.resultroot+'WARNING','w') as wp:
                            wp.write(f"WARNING: The standard deviation of offsets estimation for {key} is {z_std}. Not accurate!")
                    fixed_dict1[key] = np.mean(offset_raw,axis=0)

        for idxA, fnA in enumerate(filenamesA):
        # for idxA, fnA in tqdm(enumerate(filenamesA)):
            fnnA = fnA.split('/')[-1]
            print(fnA)
            if len(self.imgA)==num_patch:
                break

            if self.name != 'denoise':
                if self.aligned or Stitch:
                    idxB = idxA
                else:
                    idxB = random.randint(0,len(filenamesB)-1)
                fnB = filenamesB[idxB]
            else:
                if Stitch: # only for test
                    fnB = fnA
                    # fnB = fnB[0]
                elif self.aligned:
                    fnB = fdict[fnA]
                    fnB = fnB[random.randint(0,len(fnB)-1)]
                else:
                    fdict_keys = [k for k in fdict.keys()]
                    fnB = fdict[fdict_keys[random.randint(0,len(fdict_keys)-1)]]
                    fnB = fnB[random.randint(0,len(fnB)-1)]
            print(fnB)

            label_reader = AICSImage(fnA) #CZYX
            label = label_reader.data

            # AICSImage reader now gives different dimension
            while len(label.shape) > 4:
                label = np.squeeze(label,axis=0)

            if label.shape[1]<label.shape[0]:
                label = np.transpose(label,(1,0,2,3))

            input_reader = AICSImage(fnB) #CZYX
            input_img = input_reader.data

            # AICSImage reader now gives different dimension
            while len(input_img.shape) > 4:
                input_img = np.squeeze(input_img,axis=0)

            if input_img.shape[1] < input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))
            if self.name != 'align20to100':
                if len(norm_passed) > 2:
                    input_img = self.simple_norm(input_img, (norm_passed[2], norm_passed[3]))
                else:
                    input_img = self.simple_norm(input_img, (norm_passed[0], norm_passed[1]))

            #process domain B
            if self.name ==  'lr2hr' or self.name == '20to100' or self.name == 'H2B':
                if self.resizeA == 'upscale': #e.g.: size_in = (z:26,y:259,x:379); size_out = (2*z,3*y, 3*x); self.up_scale = (2,3,3)
                    self.new_size_dA = (label.shape[1],label.shape[2],label.shape[3])
                    self.new_size_dB = (label.shape[1]*self.up_scale[0],label.shape[2]*self.up_scale[1],label.shape[3]*self.up_scale[2])
                elif self.resizeA == 'ratio': #
                    try:
                        ratio_param = [float(r) for r in self.opt.ratio_param.split(',')]
                    except:
                        import sys
                        print("\nRatio parameter is required. Makse sure to provide z_ratio,y_ratio,x_ratio to ratio_param in the yaml file")
                        sys.exit(0)
                    print(ratio_param)
                    new_size = (rint(label.shape[1]*ratio_param[0]), rint(label.shape[2]*ratio_param[1]), rint(label.shape[3]*ratio_param[2]))  #100toSR
                    self.new_size_dA = new_size
                    self.new_size_dB = new_size
                elif self.resizeA == 'toB': #resize lr to hr shape
                    print(input_img.shape)
                    new_size = (input_img.shape[1],input_img.shape[2],input_img.shape[3])
                    self.new_size_dA = new_size
                    self.new_size_dB = new_size
                elif self.resizeA == 'iso': #isotropic
                    temp_new_size_lr = (rint(label.shape[1]*0.53/0.108), rint(label.shape[2]*0.27/0.108), rint(label.shape[3]*0.27/0.108))
                    temp_new_size_hr = (rint(input_img.shape[1]*0.29/0.108), input_img.shape[2], input_img.shape[3])
                    new_size = (rint(max(temp_new_size_lr[0],temp_new_size_hr[0])//2*2), rint(max(temp_new_size_lr[1],temp_new_size_hr[1])//2*2),\
                         rint(max(temp_new_size_lr[2],temp_new_size_hr[2])//2*2))
                    self.new_size_dA = new_size
                    self.new_size_dB = new_size
                elif self.resizeA == 'none':
                    self.new_size_dA = (label.shape[1],label.shape[2],label.shape[3])
                    self.new_size_dB = (label.shape[1],label.shape[2],label.shape[3])
                else: # assign a resolution by user: --resizeA z,y,x
                    try:
                        a,b,c = [int(i)  for i in self.resizeA.split(',')]
                        self.new_size_dA = (a,b,c)
                        self.new_size_dB = (a,b,c)
                    except:
                        raise NotImplementedError('resizeA should be [upscale|ratio|toB|iso|z,y,x]')
                input_img = np.expand_dims(input_img,0)
                m = Upsample(size=self.new_size_dB, mode='trilinear',align_corners=True)
                input_img = m(from_numpy(input_img)).numpy()
                input_img = np.squeeze(input_img,axis=0)
            else:
                new_size = (rint(input_img.shape[1]), rint(input_img.shape[2]), rint(input_img.shape[3]))
                self.new_size_dA = new_size
                self.new_size_dB = new_size

            #process domain A
            if self.name in ['bin2real', 'fiber', 'real2bin']:
                if self.name == 'fiber':
                    if np.sum(label)//255 < 300:
                        print('blank image')
                        continue
                #label = label//255
                # HACK: assume label has [0, 1] or [0, 255]
                label = label // label.max()
                if self.datarange_11:
                    label = label * 2.0 -1.0
            elif self.name in ['mc2real', 'clean2real', 'denoise']:
                label = self.simple_norm(label, (norm_passed[0], norm_passed[1])) # 3.5 25 for fbl and npm1
            elif self.name in ['lr2hr', '20to100', 'H2B']:
                m = Upsample(size=self.new_size_dA, mode='trilinear',align_corners=True)
                label = self.simple_norm(label, (norm_passed[0], norm_passed[1]))
                label = np.expand_dims(label,0)
                label = m(from_numpy(label)).numpy()
                label = np.squeeze(label,axis=0)
            elif self.name == 'align20to100':
                pass
            else:
                raise NotImplementedError('name should be [bin2real|mc2real|clean2real|lr2hr|fiber]')

            if (self.opt.model in ['stn'] and self.opt.stn_adjust_image):
                if self.opt.isTrain:
                    shifted_stacks_dir = self.opt.resultroot + '/shift/'
                    if not os.path.isdir(shifted_stacks_dir):
                        os.makedirs(shifted_stacks_dir)
                    if fnnA in self.stn_adjust_dict:
                        imsave(shifted_stacks_dir+f'{fnnA}_rA.tiff',label[0])
                        print(fnnA,self.stn_adjust_dict[fnnA])
                        offsets_zyx = self.stn_adjust_dict[fnnA]
                        offsets_zyx[0] = offsets_zyx[0]*1.0/self.up_scale[0]
                        offsets_zyx[1] = offsets_zyx[1]*1.0/self.up_scale[1]
                        offsets_zyx[2] = offsets_zyx[2]*1.0/self.up_scale[2]
                        with open(shifted_stacks_dir+'shift.log','a') as fp:
                            fp.write(f'{fnnA},{offsets_zyx[0]},{offsets_zyx[1]},{offsets_zyx[2]}\n')
                        offsets_zyx = from_numpy(offsets_zyx)
                        tensor = from_numpy(np.expand_dims(label,0))
                        label = self.apply_adjust(tensor,offsets_zyx)
                        label = np.squeeze(label.detach().cpu().numpy(),axis=0)
                        imsave(shifted_stacks_dir+f'{fnnA}_rA_new.tiff',label[0])
            elif self.opt.stn_adjust_fixed_z:
                if self.opt.isTrain:
                    print(f'adjust fixed z')
                    if fnnA in fixed_dict1:
                        z,y,x = fixed_dict1[fnnA]
                    else:
                        z,y,x = 0,0,0
                        raise ValueError(f'***************\n\nERROR: {fnnA} is not found in {self.opt.readoffsetfrom}, please train the AutoAlign first to get the offsets\n\n ***************\n')
                    # z = fixed_dict1_deprecated[int(fnnA[:3])]
                    if self.opt.align_all_axis:
                        offsets_zyx = np.array((z/self.up_scale[0],y/self.up_scale[1],x/self.up_scale[2]))
                    else:
                        offsets_zyx = np.array((z/self.up_scale[0],0,0))
                    offsets_zyx = from_numpy(offsets_zyx)
                    tensor = from_numpy(np.expand_dims(label,0))
                    label = self.apply_adjust(tensor,offsets_zyx)
                    label = np.squeeze(label.detach().cpu().numpy(),axis=0)
                    label = label[:,1:-1,:,:]
                    input_img = input_img[:,int(self.up_scale[0]):-int(self.up_scale[0]),:,:]
            elif self.sample_mode in ['shift', 'shift_n_shuffle', 'shift_odd','shift_first'] and (not (self.opt.model in ['stn','edsrda','rdnda'] and self.opt.stn_adjust_image)):
                if fnnA not in self.shift_dict:
                    print(f'create key in shift_dict: {fnnA}')
                    self.shift_dict[fnnA] = 0
                elif self.ToApplyShift:
                    shifted_stacks_dir = self.opt.resultroot + '/shift/'
                    imsave(shifted_stacks_dir+f'{fnnA}_rB.tiff',input_img[0])
                    imsave(shifted_stacks_dir+f'{fnnA}_rA.tiff',label[0])
                    label,input_img = self.apply_shift(label,input_img,self.shift_dict[fnnA])
                    print(f'apply {self.shift_dict[fnnA]} to image {fnnA}')
                    imsave(shifted_stacks_dir+f'{fnnA}_rB_new.tiff',input_img[0])
                    imsave(shifted_stacks_dir+f'{fnnA}_rA_new.tiff',label[0])

            if Stitch == True:
                overlap_step = 0.5
                # self.position = [(input_img.shape[1],input_img.shape[2],input_img.shape[3],)]
                self.positionB = [self.new_size_dB,]
                self.positionA = [self.new_size_dA,]
                px_list, py_list, pz_list = [], [], []
                px,py,pz = 0,0,0
                while px < self.new_size_dA[2] - self.size_in[2]:
                    px_list.append(px)
                    px+= int(self.size_in[2]*overlap_step)
                px_list.append( self.new_size_dA[2] - self.size_in[2])
                while py < self.new_size_dA[1] - self.size_in[1]:
                    py_list.append(py)
                    py+= int(self.size_in[1]*overlap_step)
                py_list.append( self.new_size_dA[1] - self.size_in[1])
                while pz < self.new_size_dA[0] - self.size_in[0]:
                    pz_list.append(pz)
                    pz+= int(self.size_in[0]*overlap_step)
                pz_list.append( self.new_size_dA[0] - self.size_in[0])
                for pz_in in pz_list:
                    for py_in in py_list:
                        for px_in in px_list:
                            (self.imgA).append(label[:,pz_in:pz_in+self.size_in[0],py_in:py_in+self.size_in[1],px_in:px_in+self.size_in[2]] )
                            (self.imgA_path).append(fnA)
                            (self.imgA_short_path).append(fnnA)
                            pz_out = pz_in * self.up_scale[0]
                            py_out = py_in * self.up_scale[1]
                            px_out = px_in * self.up_scale[2]
                            (self.imgB).append(input_img[:,pz_out:pz_out+self.size_out[0],py_out:py_out+self.size_out[1],px_out:px_out+self.size_out[2]] )
                            (self.imgB_path).append(fnB)
                            self.positionB.append((pz_out,py_out,px_out))
                            self.positionA.append((pz_in,py_in,px_in))
                            self.for_calc_ave_offset.append(False)
            else:
                #TODO: data augmentation
                new_patch_num = 0
                while new_patch_num < self.num_patch_per_img[idxA]:
                    pz = random.randint(0, label.shape[1] - self.size_in[0])
                    py = random.randint(0, label.shape[2] - self.size_in[1])
                    px = random.randint(0, label.shape[3] - self.size_in[2])
                    (self.imgA).append(label[:,pz:pz+self.size_in[0],py:py+self.size_in[1],px:px+self.size_in[2]] )
                    (self.imgA_path).append(fnA)
                    (self.imgA_short_path).append(fnnA)

                    #TODO: good crop?
                    if not self.aligned:
                        pz = random.randint(0, input_img.shape[1] - self.size_out[0])
                        py = random.randint(0, input_img.shape[2] - self.size_out[1])
                        px = random.randint(0, input_img.shape[3] - self.size_out[2])
                    else:
                        pz = pz * self.up_scale[0]
                        py = py * self.up_scale[1]
                        px = px * self.up_scale[2]
                    (self.imgB).append(input_img[:,pz:pz+self.size_out[0],py:py+self.size_out[1],px:px+self.size_out[2]] )
                    (self.imgB_path).append(fnB)
                    if new_patch_num > self.num_patch_per_img[idxA] *1//3:
                        self.for_calc_ave_offset.append(True)
                    else:
                        self.for_calc_ave_offset.append(False)
                    new_patch_num += 1
        if self.sample_mode in ['shift_n_shuffle', 'shuffle','shift_odd'] and not Stitch:
            self.imgA,self.imgB,self.imgA_path,self.imgB_path = shuffle(self.imgA,self.imgB,self.imgA_path,self.imgB_path)

        if self.name in ['real2bin','denoise']:
            self.imgA, self.imgB = self.imgB, self.imgA
            self.imgA_path,self.imgB_path = self.imgB_path,self.imgA_path


    def apply_shift(self,img1,img2,z):
        z = int(np.round(z))
        assert len(img1.shape) == 4
        assert len(img2.shape) == 4
        if z == 0:
            return img1, img2
        elif z > 0:
            new_img1 = img1[:,:-z,:,:]
            new_img2 = img2[:,z:,:,:]
        else: #z<0
            new_img1 = img1[:,-z:,:,:]
            new_img2 = img2[:,:z,:,:]
        return new_img1,new_img2

    def apply_adjust(self,tensor,offsets_zyx):
        '''
        dz>0: stack goes up (-z:-1)=padding
        dy>0: goes up; dx>0: goes right
        '''
        assert len(tensor.shape) == 5
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        tensor=tensor.to(self.device)
        dz,dy,dx = offsets_zyx.type(torch.float32)
        nz,ny,nx = tensor.shape[2:]
        z_linspace = torch.linspace(-1,1,steps=nz,device=self.device)+dz*2.0/nz     # linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        y_linspace = torch.linspace(-1,1,steps=ny,device=self.device)+dy*2.0/ny
        x_linspace = torch.linspace(-1,1,steps=nx,device=self.device)+dx*2.0/nx
        z_grid, y_grid, x_grid = torch.meshgrid(z_linspace, y_linspace, x_linspace)
        grid = torch.cat([x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), z_grid.unsqueeze(-1)], dim=-1).unsqueeze(0)
        return F.grid_sample(tensor, grid,padding_mode='border')

    def get_fdict(self, filenamesA, filenamesB):
        fdict = OrderedDict()
        fnA_dir = os.path.dirname(filenamesA[0])
        for fnB in filenamesB:
            fnB_base = os.path.basename(fnB)
            fnA = fnA_dir + '/'+ fnB_base[:7] + '_ground_truth.tiff'
            if not fnA in filenamesA:
                continue
            if fnA not in fdict:
                fdict[fnA] = [fnB,]
            else:
                fdict[fnA].append(fnB)
        print('Dict is built. Keys:')
        for key in fdict:
            print(key)
        return fdict

    def __getitem__(self, index):
        assert self.filenamesA != None, 'please load_from_file first'
        idxA = index * self.batch_size
        if idxA+self.batch_size > len(self.imgA):
            raise IndexError('end of one epoch')
        if self.aligned:
            idxB = np.array(range(idxA,idxA+self.batch_size)).tolist()
        else:
            idxB = np.random.randint(0,len(self.imgB),self.batch_size).tolist()
        image_tensorA = from_numpy(np.array(self.imgA[idxA:idxA+self.batch_size]).astype(float))
        image_tensorB = from_numpy(np.array([self.imgB[idx] for idx in idxB]).astype(float))
        # TODO: use dataloader
        return {'A': image_tensorA.float(), 'B':image_tensorB.float(), \
            'A_paths': self.imgA_path[idxA], 'B_paths': [self.imgB_path[idx] for idx in idxB][0],\
            'calcAveOffset':self.for_calc_ave_offset[idxA],'A_short_path': self.imgA_short_path[idxA]} #not support batch_size>1

    def __len__(self):
        return len(self.imgA)

    def get_size_out(self):
        return self.size_out

    def get_up_scale(self):
        return self.up_scale

    def get_num_patch_per_img(self):
         return self.num_patch_per_img