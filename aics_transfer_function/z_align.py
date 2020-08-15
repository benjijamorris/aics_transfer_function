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


all_mean = []
all_std = []
all_h =  []

def simple_norm(struct_img, scaling_param, inplace=True):
    if not inplace:
        struct_img = np.copy(struct_img)
    m, s = norm.fit(struct_img.flat)
    strech_min = max(m - scaling_param[0] * s, struct_img.min())
    strech_max = min(m + scaling_param[1] * s, struct_img.max())
    struct_img[struct_img > strech_max] = strech_max
    struct_img[struct_img < strech_min] = strech_min
    struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
    struct_img = struct_img * 2.0 - 1.0
    return struct_img.astype('float32')

def read_file(fname,do_norm=False):
    img_reader = AICSImage(fname) #CZYX
    img = img_reader.data
    if do_norm:
        img = simple_norm(img, (3.5, 15))
    img = np.squeeze(img)
    return img

def find_peaks(x):
    find_peaks = False
    for h_std in [-0.7,-0.75,-0.8]:
        pk, _ =ss.find_peaks(x,height=h_std)
        width = ss.peak_widths(x, pk, rel_height=0.5)[0]
        if len(pk) == 2 and (width[0] < 20 and width[1] < 20) and pk[0]>6:
            find_peaks = True
            break
    if not find_peaks:
        return [0,0]
    return pk

def find_peaks_deprecated(x):
    pk, _ =ss.find_peaks(x,height=-0.7)
    width = ss.peak_widths(x, pk, rel_height=0.5)[0]
    if len(pk) != 2:
        return [0,0]
    if width[0] > 20 or width[1] > 20:
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
            pdb.set_trace()
        pass
            # raise ValueError('invalid argument coord. should have two peaks')
    k,b = np.polyfit(pk1_list, pk2_list, 1)    #pk2 = k*pk1 + b
    # print(pk1_list,pk2_list)
    height_diff = analysis(pk1_list,pk2_list)
    shift = np.average([pk2_list[i]-pk1_list[i] for i in range(len(pk1_list))])
    return k,b,height_diff,shift

def torch_resize(img,new_size):
    assert len(img.shape) == 3
    img = np.expand_dims(img,0)
    img = np.expand_dims(img,0)
    m = Upsample(size=new_size, mode='trilinear',align_corners=True) #.cuda('0')
    img = m(from_numpy(img)).numpy()
    img = np.squeeze(img,axis=(0,1))
    return img

def apply_align(img1,img2,k,b):
    # keep img2 not change or only cropped
    # align img1 to img2
    # xp,xq = img1_start_z, img1_end_z
    # yp,yq = img2_start_z, img2_end_z
    # y (i.e., img2) = k * x (i.e., img1) + b
    assert img1.shape[0] == img2.shape[0], f"img1.shape={img1.shape},img2.shape={img2.shape}"
    assert len(img1.shape) == 3
    nz = img1.shape[0]
    yp = 0
    yq = nz
    xp = -b/(k+1e-6)
    xq = (nz-b)/(k+1e-6)
    union_p1 = int(max(yp,xp))
    union_q1 = int(min(yq,xq))
    img1 = img1[union_p1:union_q1,:,:]

    union_p = max(yp,xp)
    union_q = min(yq,xq)
    union_p, union_q = int(k*union_p + b), int(k* union_q+b)
    img2 = img2[union_p:union_q,:,:]

    img1 = torch_resize(img1,(img2.shape[0],img1.shape[1],img1.shape[2]))
    return img1,img2

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save(name,img):
    assert len(img.shape) == 3
    img = np.expand_dims(img,0)
    img = np.expand_dims(img,0)
    imsave(name,img)


def folder(test=True):
    folder_20x = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/20x'
    folder_100x = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/100x'
    folder_20x_iso = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/20x_iso'
    folder_100x_iso = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/100x_iso'
    folder_verify_iso = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/verify_iso'

    fname_20x_list = sorted(glob(folder_20x+'/*.tiff'))
    fname_100x_list = sorted(glob(folder_100x+'/*.tiff'))
    for i,fname_20x in enumerate(fname_20x_list):
        fname_100x = fname_100x_list[i]

        print(f'reading {fname_20x}')
        rb = read_file(fname_100x,do_norm=True)
        fb = read_file(fname_20x+'_result/fB.tiff')
        ra = read_file(fname_20x,do_norm=True)

        new_size = (int(int(rb.shape[0]*0.29/0.108)//2*2),rb.shape[1],rb.shape[2])
        # print(f'resizing rb: {rb.shape} -> {new_size}')
        rb = torch_resize(rb,new_size)
        # print(f'resizing fb: {fb.shape} -> {new_size}')
        fb = torch_resize(fb,new_size)
        # print(f'resizing ra: {ra.shape} -> {new_size}')
        ra = torch_resize(ra,new_size)

        # print(f'align')
        k,b,_,_ = get_align_kb(fb,rb)
        if not test:
            iso_a, iso_b = apply_align(ra,rb,k,b)

            print('saving')
            save(folder_20x_iso + '/' +fname_20x.split('/')[-1], iso_a)
            save(folder_100x_iso + '/' +fname_100x.split('/')[-1], iso_b)

            iso_fb, iso_rb = apply_align(fb,rb,k,b)
            save(folder_verify_iso + '/fb' +fname_100x.split('/')[-1], iso_fb)
            save(folder_verify_iso + '/rb' +fname_100x.split('/')[-1], iso_rb)
        print('__________________________')
        


def demo():
    fb = read_file('/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/20x/114_20190520_R08-Scene-26-P35-C3.tiff_result/fB.tiff')
    rb = read_file('/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/100x/114_20190520_R10-Scene-29-P35-C3.tiff',do_norm=True)
    new_size = (int(int(rb.shape[0]*0.29/0.108)//2*2),rb.shape[1],rb.shape[2])
    print(f'resizing rb: {rb.shape} -> {new_size}')
    rb = torch_resize(rb,new_size)
    print(f'resizing fb: {fb.shape} -> {new_size}')
    fb = torch_resize(fb,new_size)
    print(f'align')
    k,b,_,_ = get_align_kb(fb,rb)

def get_coord(img,index):
    five_points_dict = {110:[(143,695,26,28),
                            (303,599,33,28),
                            (210,424,31,28),
                            (243,171,13,28),
                            (544,208,12,7)],
                       111:[(397,111,21,22),
                            (377,616,25,26),
                            (505,659,17,7),
                            (392,786,31,30),
                            (84 ,289,13,18)],
                       112:[(303,495,10,16),
                            (248,386,17,9),
                            (322,280,31,28),
                            (422,787,39,30),
                            (202,818,41,15)],
                       113:[(316,804,18,31),
                            (478,669,16,10),
                            (525,215,12,11),
                            (139,123,8,17),
                            (231,20,33,31)],
                       114:[(492,625,27,21),
                            (207,699,24,21),
                            (329,191,18,19),
                            (560,123,14,15),
                            (57,85,15,13)]}
    return five_points_dict[img][index]


def folder_dev_sensitivity(test=True):
    exp = 0
    img_id = 110
    with open(f'/allen/aics/assay-dev/summer2019/Projects/seg2mito/cg3d_dev/cg3d/debug2/shift_exp_{exp}.log','a') as fp:
    	fp.write(f'---------------------------\n')
    for exp in [0,6]:
        for img_id in range(110,115):
            path_fb = glob(f'/allen/aics/assay-dev/summer2019/Projects/seg2mito/cg3d_dev/cg3d/debug2/exp_{exp}/{img_id}*/fB.tiff')[0]
            path_rb = glob(f'/allen/aics/assay-dev/summer2019/Projects/seg2mito/cg3d_dev/cg3d/debug2/exp_{exp}/{img_id}*/rB.tiff')[0]
            with open(f'/allen/aics/assay-dev/summer2019/Projects/seg2mito/cg3d_dev/cg3d/debug2/shift_exp_{exp}.log','a') as fp:
                fB = read_file(path_fb)
                rB = read_file(path_rb)
                for i in range(5):
                    coord = get_coord(img_id,i)
                    if np.round(np.min(fB)) == 0:
                        fB = fB*2-1
                        rB = rB*2 - 1
                    _,__, height_diff, shift = get_align_kb(fB,rB,coord=coord)
                    fp.write(f'{exp},{img_id},{height_diff},{shift},\n')
                    print(f'{exp},{img_id},{height_diff},{shift}')


def folder_dev_stn(test=True):
    exp = 0
    img_id = 110
    name = 'debug2'
    with open(f'/allen/aics/assay-dev/summer2019/Projects/seg2mito/cg3d_dev/cg3d/{name}/shift_exp_{exp}.log','a') as fp:
    	fp.write(f'---------------------------\n')
    for exp in range(1):
        for img_id in range(110,115):
            path_fb = glob(f'/allen/aics/assay-dev/summer2019/Projects/seg2mito/cg3d_dev/cg3d/{name}/{img_id}*/fB0.tiff')[0]
            path_rb = glob(f'/allen/aics/assay-dev/summer2019/Projects/seg2mito/cg3d_dev/cg3d/{name}/{img_id}*/rB.tiff')[0]
            with open(f'/allen/aics/assay-dev/summer2019/Projects/seg2mito/cg3d_dev/cg3d/{name}/shift_exp_{exp}.log','a') as fp:
                fB = read_file(path_fb)
                rB = read_file(path_rb)
                for i in range(5):
                    coord = get_coord(img_id,i)
                    if np.round(np.min(fB)) == 0:
                        fB = fB*2-1
                        rB = rB*2 - 1
                    _,__, height_diff, shift = get_align_kb(fB,rB,coord=coord)
                    fp.write(f'{exp},{img_id},{height_diff},{shift},\n')
                    print(f'{exp},{img_id},{height_diff},{shift}')


if __name__ == '__main__':
    folder_dev_stn()
    # folder_dev_sensitivity()

def folder_dev(test=True):
    folder_20x = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/20x'
    folder_100x = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/100x'
    folder_20x_iso = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/20x_iso'
    folder_100x_iso = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/100x_iso'
    folder_verify_iso = '/allen/aics/assay-dev/summer2019/data/lamin/paired_20x_100x/verify_iso'
    folder_20x_fb = '/allen/aics/assay-dev/summer2019/Projects/seg2mito/cycleGAN/results/0615_1441_3c51_20to100_3fc367/'

    fname_20x_list = sorted(glob(folder_20x+'/*.tiff'))
    fname_100x_list = sorted(glob(folder_100x+'/*.tiff'))
    for i,fname_20x in enumerate(fname_20x_list):
        fname_100x = fname_100x_list[i]
        fname_fb = folder_20x_fb + fname_20x.split('/')[-1]+'_result/fB.tiff'

        print(f'reading {fname_20x}')
        rb = read_file(fname_100x,do_norm=True)
        fb = read_file(fname_fb)
        ra = read_file(fname_20x,do_norm=True)

        # new_size = (int(int(rb.shape[0]*0.29/0.108)//2*2),rb.shape[1],rb.shape[2])
        new_size = rb.shape
        print(f'resizing rb: {rb.shape} -> {new_size}')
        rb = torch_resize(rb,new_size)
        print(f'resizing fb: {fb.shape} -> {new_size}')
        fb = torch_resize(fb,new_size)
        print(f'resizing ra: {ra.shape} -> {new_size}')
        ra = torch_resize(ra,new_size)

        print(f'align')
        k,b,_,_ = get_align_kb(fb,rb)
        if not test:
            iso_a, iso_b = apply_align(ra,rb,k,b)

            print('saving')
            save(folder_20x_iso + '/' +fname_20x.split('/')[-1], iso_a)
            save(folder_100x_iso + '/' +fname_100x.split('/')[-1], iso_b)

            iso_fb, iso_rb = apply_align(fb,rb,k,b)
            save(folder_verify_iso + '/fb' +fname_100x.split('/')[-1], iso_fb)
            save(folder_verify_iso + '/rb' +fname_100x.split('/')[-1], iso_rb)
        print('__________________________')

# folder_dev(test=True)
# print(all_mean,all_std,all_h)

def get_align_kb_deprecated(img1,img2,degree=1,coord=None):
    #coord = (y,x,sy,sx)
    assert img1.shape[0] == img2.shape[0], f"img1.shape={img1.shape},img2.shape={img2.shape}"
    assert len(img1.shape) == 3
    n_sample = 10
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

        if pk1[0] != 0 and pk2[0]!=0: #i.e., num of peak ==2 and width small
            pk1_list.extend(pk1)
            pk2_list.extend(pk2)
            # print(pk1,pk2,(y,x,sy,sx))
            n_sample -= 1
        elif coord is None:
            raise ValueError('invalid argument coord. should have two peaks')
    if degree == 1:
        k,b = np.polyfit(pk1_list, pk2_list, 1)    #pk2 = k*pk1 + b
    elif degree == 0:
        # print(pk1_list,pk2_list)
        analysis(pk1_list,pk2_list)
        b = np.average([pk2_list[i]-pk1_list[i] for i in range(len(pk1_list))])
        k = 1
    else:
        raise ValueError('degree should be 0 or 1')
    print(k,b)
    return k,b
