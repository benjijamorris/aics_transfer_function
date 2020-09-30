import os
from .options.base_options import BaseOptions
from .models import create_model
from .dataloader.cyclelarge_dataset import cyclelargeDataset
from .dataloader.cyclelargenopad_dataset import cyclelargenopadDataset
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

# from util.visualizer import save_images
# from util import html
import tifffile
from tifffile import imread, imsave
import shutil

from glob import glob
import pdb
import numpy as np
import random

def extract_filename(filename, replace=False, old_name= '', rep_name =''):
    filename_rev = filename[::-1]
    idx = filename_rev.index('/')
    new = filename_rev[0:idx][::-1]
    if replace:
        new = new.replace(old_name, rep_name)
    return new


def arrange(opt,data,output,position):
    data = data[0,0].cpu().numpy()
    za,ya,xa = position
    patch_size = data.shape
    z1 = 0 if za==0 else patch_size[0]//4
    z2 = patch_size[0] if za+patch_size[0]==output.shape[0] else patch_size[0]//4 + patch_size[0]//2
    y1 = 0 if ya==0 else patch_size[1]//4
    y2 = patch_size[1] if ya+patch_size[1]==output.shape[1] else patch_size[1]//4 + patch_size[1]//2
    x1 = 0 if xa==0 else patch_size[2]//4
    x2 = patch_size[2] if xa+patch_size[2]==output.shape[2] else patch_size[2]//4 + patch_size[2]//2

    # zb,yb,xb = za + opt.size_out[0], ya + opt.size_out[1], xa + opt.size_out[2]
    zaa = za+z1; zbb = za + z2
    yaa = ya+y1; ybb = ya + y2
    xaa = xa+x1; xbb = xa + x2
    output[zaa:zbb,yaa:ybb,xaa:xbb] = data[z1:z2, y1:y2, x1:x2]

def imagePSNRAndSSIM(t,v,w=9):
    m = np.max(t)-np.min(t)
    ssim = compare_ssim(t,v,multichannel=False,win_size=w,data_range=m) # if the image has only one channel, multichannel=False, win_size is an odd numher, win_size>=3
    psnr = compare_psnr(t,v,data_range=m)
    return psnr,ssim

def test():
    opt = BaseOptions(isTrain=False).parse()  # get test options
    keep_alnum = lambda s: ''.join(e for e in s if e.isalnum())
    opt.batch_size = 1    # test code only supports batch_size = 1

    if opt.name in ['real2bin','denoise']:
        opt.fpath1, opt.fpath2 = opt.fpath2, opt.fpath1

    fpath1 = opt.fpath1
    fpath2 = opt.fpath2

    if opt.testfile == 'train':
        filenamesA = (sorted(glob(fpath1+'*.tiff'),key=keep_alnum)+sorted(glob(fpath1+'*.tif'),key=keep_alnum))[:opt.train_num]
        filenamesB = (sorted(glob(fpath2+'*.tiff'),key=keep_alnum)+sorted(glob(fpath2+'*.tif'),key=keep_alnum))[:opt.train_num]
    elif opt.testfile == 'all':
        filenamesA = (sorted(glob(fpath1+'*.tiff'),key=keep_alnum)+sorted(glob(fpath1+'*.tif'),key=keep_alnum))[:]
        filenamesB = (sorted(glob(fpath2+'*.tiff'),key=keep_alnum)+sorted(glob(fpath2+'*.tif'),key=keep_alnum))[:]
    elif opt.testfile in ['test','']:
        filenamesA = (sorted(glob(fpath1+'*.tiff'),key=keep_alnum)+sorted(glob(fpath1+'*.tif'),key=keep_alnum))[opt.train_num:]
        filenamesB = (sorted(glob(fpath2+'*.tiff'),key=keep_alnum)+sorted(glob(fpath2+'*.tif'),key=keep_alnum))[opt.train_num:]
    elif os.path.isdir(opt.testfile):
        print(f'load from folder: {opt.testfile}')
        filenamesA = (sorted(glob(opt.testfile+'/*.tiff'),key=keep_alnum)+sorted(glob(opt.testfile+'/*.tif'),key=keep_alnum)+sorted(glob(opt.testfile+'/*source.tif'),key=keep_alnum)+sorted(glob(opt.testfile+'/*source.tiff'),key=keep_alnum))[:]
        filenamesB = filenamesA
        print(filenamesA)
    elif os.path.isfile(opt.testfile):
        filenamesA = [opt.testfile,]
        filenamesB = filenamesA
    else:
        raise ValueError(f'--testfile should be [train|test|all|filename|directory|]. Invalid value: {opt.testfile}')

    if 'nopad' in opt.netG:
        dataset = cyclelargenopadDataset(opt)
    else:
        dataset = cyclelargeDataset(opt,aligned=True)
    opt.size_out = dataset.get_size_out()
    opt.up_scale = dataset.get_up_scale()
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    for fileA,fileB in zip(filenamesA,filenamesB):
        dataset.load_from_file([fileA,], [fileB,], num_patch=-1)
        position = dataset.positionB
        positionA = dataset.positionA
        rA = np.zeros(positionA[0]).astype('float32')
        rB = np.zeros(position[0]).astype('float32')
        fA = np.zeros(positionA[0]).astype('float32')
        fB = np.zeros(position[0]).astype('float32')
        fB0 = np.zeros(position[0]).astype('float32')
        recA = np.zeros(positionA[0]).astype('float32')
        recB = np.zeros(position[0]).astype('float32')
        ffB = np.zeros(position[0]).astype('float32')
        rrecB = np.zeros(position[0]).astype('float32')

        print(position)
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            if opt.output_path == 'model':
                prefix = opt.continue_from + '/' +  fileA.split('/')[-1] + '_result/'
            elif opt.output_path == 'data':
                prefix = fileA + '_result/'
            else:
                prefix = opt.output_path + '/' +  fileA.split('/')[-1] + '_result/'
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            shutil.copy(opt.config,prefix)

            za,ya,xa = position[i+1]
            print((za,ya,xa))
            zb,yb,xb = za + opt.size_out[0], ya + opt.size_out[1], xa + opt.size_out[2]
            print((zb,yb,xb))

            if opt.model == 'pix2pix':
                rA_i, rB_i, fB_i = model.test()           # run inference
                # psnr_list.append(psnr.psnr_local(rB_i[0,0].cpu().numpy(),fB_i[0,0].cpu().numpy()))
                arrange(opt,rA_i,rA,positionA[i+1])
                arrange(opt,rB_i,rB,position[i+1])
                arrange(opt,fB_i,fB,position[i+1])
            elif  opt.model in ['stn']:
                rA_i, rB_i, fB0_i, fB_i = model.test()           # run inference
                # psnr_list.append(psnr.psnr_local(rB_i[0,0].cpu().numpy(),fB0_i[0,0].cpu().numpy()))
                arrange(opt,rA_i,rA,positionA[i+1])
                arrange(opt,rB_i,rB,position[i+1])
                arrange(opt,fB0_i,fB0,position[i+1])
                arrange(opt,fB_i,fB,position[i+1])
 
        if opt.resizeA == 'upscale':
            from torch import from_numpy
            from  torch.nn.modules.upsampling import Upsample
            from aicsimageio import AICSImage
            input_reader = AICSImage(fileB) #CZYX
            new_size = input_reader.data.shape[2:]
            op = Upsample(size=new_size, mode='trilinear',align_corners=True)
            def resize_(img,op):
                img = np.expand_dims(img,0)
                img = np.expand_dims(img,0)
                img = op(from_numpy(img)).numpy()
                img = np.squeeze(img,axis=0)
                img = np.squeeze(img,axis=0)
                return img
            rB = resize_(rB,op)
            fB = resize_(fB,op)

        # if opt.model in ['stn','pix2pix','rdn','rdnd','edsr','edsrd','edsrda','rdnda'] and opt.name == '20to100':
        #     with open(prefix+'../height_error.csv','a') as fp:
        #         img_id = int(fileA.split('/')[-1][:3])
        #         for i in range(10):
        #             coord = get_coord(img_id,i)
        #             if np.round(np.min(fB)) == 0:
        #                 fB = fB*2 - 1
        #                 rB = rB*2 - 1
        #             _,__, height_diff, shift = get_align_kb(fB,rB,coord=coord)
        #             fp.write(f'{opt.tag},{img_id},{height_diff},{shift},\n')
        #             print(f'{opt.tag},{img_id},{height_diff},{shift}')
        #     with open(prefix+'../psnr_ssim.csv','a') as fp:
        #         img_id = int(fileA.split('/')[-1][:3])
        #         psnr_,ssim = imagePSNRAndSSIM(rB,fB)
        #         fp.write(f'{opt.tag},{img_id},{psnr_},{ssim}\n')


        # imsave(prefix+'rA.tiff',rA)
        # imsave(prefix+'rB.tiff',rB)
        # imsave(prefix+'fB.tiff',fB)
        # psnr_,ssim = imagePSNRAndSSIM(rB,fB)
        # print(f'psnr: {psnr_}\nssim: {ssim}')


        ###########################
        # Temp saving script
        filename_ori = extract_filename(fileA, replace=True, old_name='source.tif', rep_name='pred.tif')
        #from aicsimageio import omeTifWriter
        #writer = omeTifWriter.OmeTifWriter(opt.output_path + "/" + filename_ori)
        #writer.save(fB)
        tif = tifffile.TiffWriter(opt.output_path + "/" + filename_ori, bigtiff=True)
        tif.save(fB, compress=9, photometric='minisblack', metadata=None)
        tif.close()
        #imsave(opt.output_path + "/" + filename_ori,fB)
        print(filename_ori + " saved")
        ###########################

        # print(f'local_psnr {np.mean(psnr_list)}')

        # imsave(prefix+'fB_yz.tiff',np.transpose(fB,(1,0,2)))
        # imsave(prefix+'fB_xz.tiff',np.transpose(fB,(2,1,0)))
        # imsave(prefix+'rB_yz.tiff',np.transpose(rB,(1,0,2)))
        # imsave(prefix+'rB_xz.tiff',np.transpose(rB,(2,1,0)))
        # imsave(prefix+'rA_yz.tiff',np.transpose(rA,(1,0,2)))
        # imsave(prefix+'rA_xz.tiff',np.transpose(rA,(2,1,0)))

        # if opt.model == 'cycle_gan':
        #     imsave(prefix+'fA.tiff',fA)
        #     imsave(prefix+'recA.tiff',recA)
        #     imsave(prefix+'recB.tiff',recB)
        # if opt.model == 'noise_gan':
        #     imsave(prefix+'ffB.tiff',ffB)
        #     imsave(prefix+'rrecB.tiff',rrecB)
        #     imsave(prefix+'fA.tiff',fA)
        #     imsave(prefix+'recA.tiff',recA)
        #     imsave(prefix+'recB.tiff',recB)
        # if opt.model == 'stn':
        #     imsave(prefix+'fB0.tiff',fB0)
        #     # print(f'global_psnr {psnr.psnr_local(rB,fB0)}')
        # if opt.model in ['pix2pix', 'rdn', 'rdnd','edsr','edsrd']:
        #     pass
        #     # print(f'global_psnr {psnr.psnr_local(rB,fB)}')
        # print('--------------')


from aicsimageio import AICSImage
import scipy.signal as ss
# find_peaks,peak_widths
from scipy.stats import norm
import numpy as np
import random
from torch import from_numpy
from  torch.nn.modules.upsampling import Upsample
from tifffile import imread, imsave
from glob import glob
import os
import pdb
from sklearn.mixture import GaussianMixture as GMM


all_mean = []
all_std = []
all_h =  []

def find_peaks(x):
    ys = np.array(x)
    ys = ys-ys.min()
    ys = ys/np.sum(ys)

    mc_sample = [np.random.choice(np.arange(len(x)), p=ys) for _ in range(100000)]
    # explaination for 100000: We use Monte Carlo method to estimate the peak position;
    # 100000 is the number of sample for MC.
    # precistion of MC = C/sqrt(N) , where N is the number of sample, C is a const number.
    # With 100000, the precistion of peak position estimation is < 0.1 voxel
    mc_sample = np.array(mc_sample)
    mc_sample = np.expand_dims(mc_sample,-1)
    clf = GMM(n_components=2).fit(mc_sample)
    pk1,pk2 = clf.means_
    if pk2<pk1:
        pk1, pk2 = pk2,pk1
    return [float(pk1),float(pk2)]

def find_peaks_deprecated(x):
    find_peaks = False
    for h_std in [-0.50,-0.55,-0.60,-0.65,-0.7,-0.75,-0.8]:
        pk, _ =ss.find_peaks(x,height=h_std)
        width = ss.peak_widths(x, pk, rel_height=0.5)[0]
        if len(pk) == 2 and (width[0] < 20 and width[1] < 20) and pk[0]>6:
            find_peaks = True
            break
    if not find_peaks:
        return [0,0]
    return pk

def get_profile(img,coord):
    (y,x,h,w) = coord
    stack = img[:,y:y+h,x:x+w]
    return np.mean(stack,axis=(1,2))

def analysis(l1,l2):
    n = len(l1)//2
    lh = []
    h2s = []
    for i in range(n):
        h1 = l1[2*i+1] - l1[2*i]
        h2 = l2[2*i+1] - l2[2*i]
        lh.append(h2-h1)
        h2s.append(h2)
    all_mean.append(np.mean(lh))
    all_std.append(np.std(lh))
    all_h.append(np.mean(h2))
    print(lh)
    print(f'mean: {np.mean(lh)};\n std: {np.std(lh)}')
    return np.mean(lh)

def get_align_kb(img1,img2,coord=None):
    #coord = (y,x,sy,sx)
    assert img1.shape[0] == img2.shape[0], f"img1.shape={img1.shape},img2.shape={img2.shape}"
    assert len(img1.shape) == 3
    n_sample = 10
    if coord is not None:
        n_sample = 1
    sy = 40
    sx = 40
    pk1_list = []
    pk2_list = []
    while n_sample != 0:
        if coord is None:
            y = random.randint(0,img1.shape[1]-sy-1)
            x = random.randint(0,img1.shape[2]-sx-1)
        else:
            (y,x,sy,sx) = coord
        p1 = get_profile(img1, (y,x,sy,sx))
        p2 = get_profile(img2, (y,x,sy,sx))
        pk1 = find_peaks(p1)
        pk2 = find_peaks(p2)

        if pk1[0] != 0 and pk2[0]!=0 and abs((pk1[1]-pk1[0])-(pk2[1]-pk2[0])) < 7: #i.e., two sharp peaks, the height difference should not be too large
            pk1_list.extend(pk1)
            pk2_list.extend(pk2)
            # print(pk1,pk2,(y,x,sy,sx))
            n_sample -= 1
        elif coord is not None:
            print(coord)
            print(ss.find_peaks(p1,height=-0.8))
            print(ss.find_peaks(p2,height=-0.8))
            # imsave('img1.tiff',img1)
            print('peaks can not be automatically found. Insert the peak values manually: (e.g. pk1=[15,32])')
            pdb.set_trace()
        pass
            # raise ValueError('invalid argument coord. should have two peaks')
    k,b = np.polyfit(pk1_list, pk2_list, 1)    #pk2 = k*pk1 + b
    # print(pk1_list,pk2_list)
    height_diff = analysis(pk1_list,pk2_list)
    shift = np.average([pk2_list[i]-pk1_list[i] for i in range(len(pk1_list))])
    return k,b,height_diff,shift


def get_coord(img,index):
    five_points_dict = {100:[(74,672,24,26),
                             (165,486,23,21),
                             (462,161,23,21),
                             (436,812,23,21),
                             (142,55,23,21) ],
                       110:[(143,695,26,28),
                            (303,599,33,28),
                            (210,424,31,28),
                            (243,171,13,28),
                            (544,208,12,7),
                            (595,595,18,18),
                            (500,804,29,28),
                            (520,52, 19,15),
                            (376,310,23,30),
                            (119,385,24,19),],
                       111:[(397,111,21,22),
                            (377,616,25,26),
                            (505,659,17,7),
                            (392,786,31,30),
                            (84 ,289,13,18),
                            (104,804,22,26),
                            (260,858,24,18),
                            (574,122,22,28),
                            (344,248,30,30),
                            (560,516,30,26)],
                       112:[(303,495,10,16),
                            (248,386,17,9),
                            (322,280,31,28),
                            (422,787,39,30),
                            (202,818,41,15),
                            (348,608,30,26),
                            (294,712,24,26),
                            (512,578,22,24),
                            (542,356,20,26),
                            (438,162,22,20)],
                       113:[(316,804,18,31),
                            (478,669,16,10),
                            (525,215,12,11),
                            (139,123,8,17),
                            (231,20,33,31),
                            (100,96,32,36),
                            (198,402,26,28),
                            (184,710,24,22),
                            (576,672,26,22),
                            (228,430,22,30)],
                       114:[(492,625,27,21),
                            (207,699,24,21),
                            (329,191,18,19),
                            (560,123,14,15),
                            (57, 85, 15,13),
                            (194,658,30,30),
                            (572,94, 20,28),
                            (142,510,28,26),
                            (66, 320,24,30),
                            (106,772,32,32)]}
    return five_points_dict[img][index]


if __name__ == '__main__':
    test()


