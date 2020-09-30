import numpy as np
import os
from tifffile import imread, imsave
from PIL import Image
import random
from tqdm import tqdm

from torch import from_numpy
from  torch.nn.modules.upsampling import Upsample
from aicsimageio import AICSImage

import time
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

import pdb

import os.path
from PIL import Image
import random

import numpy as np
from scipy.stats import norm
from aicsimageio import AICSImage
from tifffile import imread, imsave
import os
import torch

from collections import OrderedDict
from sklearn.utils import shuffle



class cyclelargenopadDataset(Dataset):
    def __init__(self, opt, aligned=False):
        Dataset.__init__(self)
        self.imgA = []
        self.imgB = []
        self.size_in = opt.size_in
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

        if self.model in ['rdn','edsr','edsrd']: #e.g.: size_in = (z:26,y:259,x:379); size_out = (2*z,3*y, 3*x); self.up_scale = (2,3,3)
            self.up_scale = (int(np.ceil(0.53/0.29)),int(np.ceil(0.27/0.108)),int(np.ceil(0.27/0.108)))
            self.size_out = (int(self.size_in[0]*self.up_scale[0]), int(self.size_in[1]*self.up_scale[1]), int(self.size_in[2]*self.up_scale[2]))
            self.resizeA = 'upscale'
        elif self.model == 'pix2pix' and self.netG=='unet_512_nopad':
            self.up_scale = (1,1,1)
            self.size_out = (self.size_in[0]-4,152,152) #size_in = (32,512,512) --> (28,152,152)
        else:
            self.up_scale = (1,1,1)
            self.size_out = self.size_in
        
        if opt.model == 'pix2pix' or opt.model in ['rdn','edsr','edsrd']:
            self.aligned = True

    def simple_norm(self, struct_img, scaling_param, inplace=True):
        if not inplace:
            struct_img = np.copy(struct_img)
        print(f'intensity normalization: normalize into [mean - {scaling_param[0]} x std, mean + {scaling_param[1]} x std] ')
        m, s = norm.fit(struct_img.flat)
        strech_min = max(m - scaling_param[0] * s, struct_img.min())
        strech_max = min(m + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
        struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
        return struct_img.astype('float32')

    def load_from_file(self, filenamesA, filenamesB, num_patch):
        rint = lambda x: int(np.round(x))
        print('load %d patches'%num_patch)
        self.imgA = []
        self.imgB = []

        if self.name != 'denoise':
            assert len(filenamesA) == len(filenamesB)
        else:
            fdict = self.get_fdict(filenamesA, filenamesB)
        num_data = len(filenamesA)
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
            label = np.squeeze(label,axis=0)
            if label.shape[1]<label.shape[0]:
                label = np.transpose(label,(1,0,2,3))

            input_reader = AICSImage(fnB) #CZYX
            input_img = input_reader.data
            input_img = np.squeeze(input_img,axis=0)
            if input_img.shape[1] < input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))
            if self.name != 'align20to100':
                input_img = self.simple_norm(input_img, (3.5, 15))
            
            if self.sample_mode == 'shift':
                if fnnA not in self.shift_dict:
                    print(f'create key in shift_dict: {fnnA}')
                    self.shift_dict[fnnA] = 0
                else:
                    label,input_img = self.apply_shift(label,input_img,self.shift_dict[fnnA])

            #process domain B
            if self.name ==  'lr2hr' or self.name == '20to100' or self.name == 'H2B':
                if self.resizeA == 'upscale': #e.g.: size_in = (z:26,y:259,x:379); size_out = (2*z,3*y, 3*x); self.up_scale = (2,3,3)
                    self.new_size_dA = (label.shape[1],label.shape[2],label.shape[3])
                    self.new_size_dB = (label.shape[1]*self.up_scale[0],label.shape[2]*self.up_scale[1],label.shape[3]*self.up_scale[2])
                elif self.resizeA == 'ratio': #
                    new_size = (rint(label.shape[1]*0.53/0.29), rint(label.shape[2]*0.27/0.108), rint(label.shape[3]*0.27/0.108))
                    self.new_size_dA = new_size
                    self.new_size_dB = new_size
                elif self.resizeA == 'toB': #resize lr to hr shape
                    # new_size = (rint(label.shape[1]*0.53/0.29), rint(label.shape[2]*0.27/0.108), rint(label.shape[3]*0.27/0.108))
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
                label = label//255 * 2.0 - 1.0 # 4-D after squeeze
            elif self.name in ['mc2real', 'clean2real', 'denoise']:
                label = self.simple_norm(label, (3.5, 15))
            elif self.name in ['lr2hr', '20to100', 'H2B']:
                m = Upsample(size=self.new_size_dA, mode='trilinear',align_corners=True)
                label = self.simple_norm(label, (3.5, 15))
                label = np.expand_dims(label,0)
                label = m(from_numpy(label)).numpy()
                label = np.squeeze(label,axis=0)
            elif self.name == 'align20to100':
                pass
            else:
                raise NotImplementedError('name should be [bin2real|mc2real|clean2real|lr2hr|fiber]')

            label = self.pad(label, self.size_in, self.size_out)
            if Stitch == True:
                overlap_step = 0.5
                # self.position = [(input_img.shape[1],input_img.shape[2],input_img.shape[3],)]
                self.position = [self.new_size_dB,]
                px_list, py_list, pz_list = [], [], []
                px,py,pz = 0,0,0
                while px < self.new_size_dB[2] - self.size_out[2]:
                    px_list.append(px)
                    px+= int(self.size_out[2]*overlap_step)
                px_list.append( self.new_size_dB[2] - self.size_out[2])
                while py < self.new_size_dB[1] - self.size_out[1]:
                    py_list.append(py)
                    py+= int(self.size_out[1]*overlap_step)
                py_list.append( self.new_size_dB[1] - self.size_out[1])
                while pz < self.new_size_dB[0] - self.size_out[0]:
                    pz_list.append(pz)
                    pz+= int(self.size_out[0]*overlap_step)
                pz_list.append( self.new_size_dB[0] - self.size_out[0])
                for pz in pz_list:
                    for py in py_list:
                        for px in px_list:
                            (self.imgB).append(input_img[:,pz:pz+self.size_out[0],py:py+self.size_out[1],px:px+self.size_out[2]] )
                            self.position.append((pz,py,px))
                            pz = pz * self.up_scale[0]
                            py = py * self.up_scale[1]
                            px = px * self.up_scale[2]
                            (self.imgA).append(label[:,pz:pz+self.size_in[0],py:py+self.size_in[1],px:px+self.size_in[2]] )

            else:
                #TODO: data augmentation
                new_patch_num = 0
                while new_patch_num < self.num_patch_per_img[idxA]:
                    pz = random.randint(0, label.shape[1] - self.size_in[0])
                    py = random.randint(0, label.shape[2] - self.size_in[1])
                    px = random.randint(0, label.shape[3] - self.size_in[2])
                    (self.imgA).append(label[:,pz:pz+self.size_in[0],py:py+self.size_in[1],px:px+self.size_in[2]] )

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
                    new_patch_num += 1
        if self.sample_mode in ['shift_n_shuffle', 'shuffle'] and not Stitch:
            self.imgA,self.imgB = shuffle(self.imgA,self.imgB)

        if self.name in ['real2bin','denoise']:
            self.imgA, self.imgB = self.imgB, self.imgA
        
    def pad(self,img,s_in,s_out):
        new_img = np.zeros((img.shape[0],img.shape[1] + s_in[0]-s_out[0],\
            img.shape[2] + s_in[1]-s_out[1], img.shape[3] + s_in[2] - s_out[2]))
        z0 = (s_in[0]-s_out[0])//2
        z1 = z0 + img.shape[1]
        y0 = (s_in[1]-s_out[1])//2
        y1 = y0 + img.shape[2]
        x0 = (s_in[2]-s_out[2])//2
        x1 = x0 + img.shape[3]
        new_img[:,z0:z1,y0:y1,x0:x1] = img
        return new_img


    def apply_shift(self,img1,img2,z):
        z = int(np.round(z))
        if z == 0:
            return img1, img2
        elif z > 0:
            # new_img[0,z:,:,:] = img[0,:-z,:,:]
            new_img1 = img1[:,:-z,:,:]
            new_img2 = img2[:,z:,:,:]
        else: #z<0
            # new_img[0,:z,:,:] = img[0,-z:,:,:]
            new_img1 = img1[:,-z:,:,:]
            new_img2 = img2[:,:z,:,:]
        return new_img1,new_img2

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
        return {'A': image_tensorA.float(), 'B':image_tensorB.float(), 'A_paths': '', 'B_paths': ''}

    def __len__(self):
        return len(self.imgA)

    def get_size_out(self):
        return self.size_out

    def get_up_scale(self):
        return self.up_scale



