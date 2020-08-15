from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
# from skimage.io import imread
from tifffile import imread
from skimage import data,img_as_float
import numpy as np
from scipy.stats import norm
from  torch.nn.modules.upsampling import Upsample
from torch import from_numpy
from glob import glob
import pdb

def shift(img,z):
    new_img = -np.ones(img.shape,dtype=float)
    if z > 0:
#         new_img[z:,:,:] = img[:-z,:,:]
        new_img = img[:-z,:,:]
    elif z==0:
        new_img = img
    else: #z<0
#         new_img[:z,:,:] = img[-z:,:,:]
        new_img = img[-z:,:,:]
    return new_img

def upsample(img,scale=None,new_size=None,mode='trilinear'):
    '''
    mode = ``'nearest'``,``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
    '''
    if new_size is None:
        new_size = (int(img.shape[0]*scale),int(img.shape[1]*scale),int(img.shape[2]*scale),)
    align_corners = True
    if mode == 'nearest':
        align_corners = None
    m = Upsample(size=new_size, mode=mode,align_corners=align_corners)
    img = np.expand_dims(img,0)
    img = np.expand_dims(img,0)
    img = m(from_numpy(img)).numpy()
    img = np.squeeze(img)
    return img
    
def simple_norm(struct_img, scaling_param, inplace=True):
    if not inplace:
        struct_img = np.copy(struct_img)
    print(f'intensity normalization: normalize into [mean - {scaling_param[0]} x std, mean + {scaling_param[1]} x std] ')
    m, s = norm.fit(struct_img.flat)
    strech_min = max(m - scaling_param[0] * s, struct_img.min())
    strech_max = min(m + scaling_param[1] * s, struct_img.max())
    struct_img[struct_img > strech_max] = strech_max
    struct_img[struct_img < strech_min] = strech_min
    struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
    struct_img = struct_img * 2.0 - 1.0
    return struct_img.astype('float32')

def imagePSNRAndSSIM(file1,file2,i,ssim=True,w=9):
    if i>0:
        t = file1[i:,:,:]
    elif i==0:
        t = file1
    else:
        t= file1[:i,:,:]
    v = file2 # generated image
    m = np.max(t)-np.min(t) 
#     print(t.shape,v.shape)
    if ssim:
        ssim = compare_ssim(t,v,multichannel=False,win_size=w,data_range=m) # if the image has only one channel, multichannel=False, win_size is an odd numher, win_size>=3
    else:
        ssim = 0
    psnr = compare_psnr(t,v,data_range=m)
    return psnr,ssim

def psnr_wholestack():
    import sys

    j = int(sys.argv[1])
    original = imread(glob(r"/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/100x/%d*.tiff"%j)[0])
    original = simple_norm(original, (3.5, 15))
    contrastfB_ori = imread(str(sys.argv[2]))

def psnr_local(original,contrastfB_ori):
    best = 0
    besti = 0
    for i in range(-5,5):
        contrastfB = shift(contrastfB_ori,i)
        psnr, _ = imagePSNRAndSSIM(original,contrastfB,i,ssim=False)
        if psnr>best:
            best = psnr
            besti = i
    print(besti,best)
    return best
    

# size = original.shape
# size_in = ()
# position = [size,]
# px_list, py_list, pz_list = [], [], []
# px,py,pz = 0,0,0
# while px < size[2] - size_in[2]:
#     px_list.append(px)
#     px+= int(size_in[2]*overlap_step)
# px_list.append( size[2] - size_in[2])
# while py < size[1] - size_in[1]:
#     py_list.append(py)
#     py+= int(size_in[1]*overlap_step)
# py_list.append( size[1] - size_in[1])
# while pz < size[0] - size_in[0]:
#     pz_list.append(pz)
#     pz+= int(size_in[0]*overlap_step)
# pz_list.append( size[0] - size_in[0])
# for pz in pz_list:
#     for py in py_list:
#         for px in px_list:
#             (imgA).append(label[:,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] )
#             pz = pz * up_scale[0]
#             py = py * up_scale[1]
#             px = px * up_scale[2]
#             (imgB).append(input_img[:,pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]] )
#                             position.append((pz,py,px))