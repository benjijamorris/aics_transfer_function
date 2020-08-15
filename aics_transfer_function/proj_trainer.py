"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from transfer_function.options.base_options import BaseOptions
from transfer_function.dataloader.cyclelarge_dataset import cyclelargeDataset
from transfer_function.dataloader.cyclelargenopad_dataset import cyclelargenopadDataset
from transfer_function.models.__init__ import create_model
from glob import glob
import pdb
from tifffile import imread, imsave
import random
from os.path import isfile
from transfer_function.util.z_align import get_align_kb
import numpy as np
import torch
import os

def save(name,data):
    data = data[0,0].cpu().numpy()
    imsave(name,data)

def get_filenames(p1,p2,s1=None,s2=None,check_name=False):
    keep_alnum = lambda s: ''.join(e for e in s if e.isalnum())
    all1 =  sorted(glob(p1+'*.tiff'),key=keep_alnum) + sorted(glob(p1+'*.tif'),key=keep_alnum)
    all2 =  sorted(glob(p2+'*.tiff'),key=keep_alnum) + sorted(glob(p2+'*.tif'),key=keep_alnum)
    if check_name:
        assert len(all1) == len(all2)
        for i in range(len(all1)):
            if s1 is not None:
                assert all1[i].split('/')[-1][s1[0]:s1[1]] == all2[i].split('/')[-1][s2[0]:s2[1]], \
                    f"Filename mismatch: {all1[i]}, {all2[i]}"
            else:
                assert all1[i].split('/')[-1] == all2[i].split('/')[-1], \
                    f"Filename mismatch: {all1[i]}, {all2[i]}"
    return all1,all2

def arrange(opt,data,output,position):
    data = data[0,0].cpu().numpy()
    za,ya,xa = position
    z1 = 0 if za==0 else opt.size_out[0]//4
    z2 = opt.size_out[0] if za+opt.size_out[0]==output.shape[0] else opt.size_out[0]//4 + opt.size_out[0]//2
    y1 = 0 if ya==0 else opt.size_out[1]//4
    y2 = opt.size_out[1] if ya+opt.size_out[1]==output.shape[1] else opt.size_out[1]//4 + opt.size_out[1]//2
    x1 = 0 if xa==0 else opt.size_out[2]//4
    x2 = opt.size_out[2] if xa+opt.size_out[2]==output.shape[2] else opt.size_out[2]//4 + opt.size_out[2]//2

    # zb,yb,xb = za + opt.size_out[0], ya + opt.size_out[1], xa + opt.size_out[2]
    zaa = za+z1; zbb = za + z2
    yaa = ya+y1; ybb = ya + y2
    xaa = xa+x1; xbb = xa + x2
    output[zaa:zbb,yaa:ybb,xaa:xbb] = data[z1:z2, y1:y2, x1:x2]

def train():
    opt = BaseOptions(isTrain=True).parse()
    if opt.seed>0:
        print(f'set seed as {opt.seed}')
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    adjust_dict = {}  # for stn adjust_dict
    epoch_for_stage_index = 0  # for stn adjust_dict
    first_adjust = True

    if opt.debug:
        opt.imgs_per_epoch = 3
        opt.patches_per_epoch = 20
        opt.train_num = 5
        print('Debug mode is on. Read fewer data.\nset opt.imgs_per_epoch = 3\n    opt.patches_per_epoch = 20\n    opt.train_num = 5 ')

    if opt.name in ['real2bin','denoise']:
        opt.fpath1, opt.fpath2 = opt.fpath2, opt.fpath1

    # creat dataset class
    if 'nopad' in opt.netG:
        dataset = cyclelargenopadDataset(opt)
    else:
        dataset = cyclelargeDataset(opt)
    opt.size_out = dataset.get_size_out()
    opt.up_scale = dataset.get_up_scale()

    all_filenamesA, all_filenamesB = get_filenames(opt.fpath1,opt.fpath2,None,None,check_name=opt.check_name)
    filenamesA = all_filenamesA[:opt.train_num]
    print('The number of training images = %d' % len(filenamesA))

    if opt.name == 'denoise':
        filenamesB = all_filenamesB
    else:
        filenamesB = all_filenamesB[:opt.train_num]

    if opt.imgs_per_epoch > min(len(filenamesA),len(filenamesB)):
        opt.imgs_per_epoch = min(len(filenamesA),len(filenamesB))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    odd_epoch_flag = True               # T -> F -> T -> F -> T....
    first_epoch_flag = True             # T -> F -> F -> F -> F....

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        print('reload data begin')
        ToApplyShift = False

        if (opt.sample_mode in ['shift', 'shift_n_shuffle','shift_odd','shift_first'] and (epoch >= 5 or opt.continue_train)) and (not(opt.model in ['stn','edsrda','rdnda'] and opt.stn_adjust_image)):
            shifted_stacks_dir = opt.resultroot + '/shift/'
            if not os.path.isdir(shifted_stacks_dir):
                os.makedirs(shifted_stacks_dir)
            if opt.sample_mode in ['shift', 'shift_n_shuffle']:
                ToApplyShift = True
            elif opt.sample_mode in ['shift_odd',]:
                if not odd_epoch_flag:
                    ToApplyShift = True
            elif opt.sample_mode in ['shift_first',]:
                if first_epoch_flag:
                    first_epoch_flag = False
                    ToApplyShift = True
            for i in range(len(filenamesA)):
                fileA = filenamesA[i]
                fileB = filenamesB[i]
                dataset.load_from_file([fileA,],[fileB,],-1)
                position = dataset.position
                rA = np.zeros(position[0]).astype('float32')
                rB = np.zeros(position[0]).astype('float32')
                fB = np.zeros(position[0]).astype('float32')
                for j,data in enumerate(dataset):
                    model.set_input(data)
                    if opt.model == 'pix2pix':
                        rA_j, rB_j, fB_j = model.test()
                        arrange(opt,fB_j,fB,position[j+1])
                    elif opt.model == 'stn':
                        rA_j, rB_j, fB0_j, _ = model.test()
                        arrange(opt,fB0_j,fB,position[j+1])
                    arrange(opt,rA_j,rA,position[j+1])
                    arrange(opt,rB_j,rB,position[j+1])
                _, offset = get_align_kb(fB, rB, degree=0)
                dataset.shift_dict[fileA.split('/')[-1]] = offset
                if True: # only for debug, dataset will be renew after load_from_file. applied shift are discarded
                    fB_new,rB = dataset.apply_shift(np.expand_dims(fB,0),np.expand_dims(rB,0),offset)
                    imsave(shifted_stacks_dir+f'{fileA.split("/")[-1][:3]}_fB_new.tiff',fB_new[0])
                    imsave(shifted_stacks_dir+f'{fileA.split("/")[-1][:3]}_rB.tiff',rB[0])
            minfileA = 0
            if opt.sample_mode in ['shift_odd'] and (not ToApplyShift):
                #find the min offset image as standard
                min_offset = 999999 #init a large value
                for i in range(len(filenamesA)):
                    fileA = filenamesA[i]
                    if abs(dataset.shift_dict[fileA.split('/')[-1]]) < min_offset:
                        min_offset =  abs(dataset.shift_dict[fileA.split('/')[-1]])
                        minfileA = i
            with open(shifted_stacks_dir + '/shift.log','a') as shift_log:
                for key in dataset.shift_dict:
                    shift_log.write(f'{key[:3]},{dataset.shift_dict[key]}\n')
                shift_log.write(f'best align {filenamesA[minfileA]}')
                shift_log.write(f'shift applied' if ToApplyShift else 'shift Not applied')
                shift_log.write('---------------------\n')

        if opt.model in ['stn']:
            opt.stn_first_stage = str(opt.stn_first_stage)
            opt.stn_loop_stage = str(opt.stn_loop_stage)
            if epoch_for_stage_index >= len(opt.stn_first_stage):
                model.stage = int((opt.stn_loop_stage)[((epoch_for_stage_index-len(opt.stn_first_stage)))%(len(opt.stn_loop_stage))])
            else:
                model.stage = int((opt.stn_first_stage)[epoch_for_stage_index])
            epoch_for_stage_index += 1

            if opt.stn_progressive_adjust: # the adjust is based on the previous adjusts. It is a training strategy
                if model.stage == 2: # the current model stage == 2
                    adjust_dict  =  {}
            else:
                if opt.stn_adjust_image and model.stage != 2:  # the adjusts are independent. It is used to verify the performance of AutoAlign module
                    for fnA in adjust_dict:
                        fnnA = fnA.split('/')[-1]
                        adjust_fnA = np.array(adjust_dict[fnA])
                        num_data = int(adjust_fnA.shape[0]*1//3)
                        adjust_mean_zyx = np.mean(adjust_fnA[num_data:,:],axis=0)
                        dataset.stn_adjust_dict[fnnA] = adjust_mean_zyx
                        print(fnnA,adjust_mean_zyx)
                else:
                    # adjust_dict: {filenameA: [shift1, shift2, ...]; filenameB ... }
                    # dataset.stn_adjust_dict: {filenameA: (z,y,x);filenameB: ...}
                    dataset.stn_adjust_dict = {}
                    adjust_dict  =  {}

        if opt.sample_mode in ['shift_odd',] and (not ToApplyShift):
            idxA = [minfileA,]
        else:
            idxA = random.sample(range(len(filenamesA)), opt.imgs_per_epoch)
            idxB = random.sample(range(len(filenamesB)), opt.imgs_per_epoch)

        if opt.model in ['pix2pix','stn']:
            idxB = idxA
        fileA = [filenamesA[i] for i in idxA]
        if opt.name  != 'denoise':
            fileB = [filenamesB[i] for i in idxB]
        else:
            fileB = filenamesB
        dataset.ToApplyShift = ToApplyShift
        dataset.load_from_file(fileA, fileB, num_patch=opt.patches_per_epoch)
        print('reload data finish\n')


        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                model.set_input(dataset[i])  # unpack data from data loader
                if opt.model in ['pix2pix']:
                    rA, rB, fB = model.test()           # run inference
                    if opt.save_training_process:
                        prefix = opt.sample_dir + str(total_iters)
                        save(prefix+'fB_pre.tiff',fB)

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if opt.model in ['stn'] and opt.stn_adjust_image:
                if model.stage == 2:
                    fnA = data['A_paths']
                    shift_zyx = model.get_shift().cpu().detach().numpy()
                    if fnA in adjust_dict:
                        adjust_dict[fnA].append(shift_zyx)
                    else:
                        adjust_dict[fnA]=[shift_zyx,]

            if total_iters == 1:
                print(data['A'].shape)
                print(data['B'].shape)

            losses = model.get_current_losses()
            message = '(epoch: %d, iters: %d, data: %.3f) ' % (epoch, total_iters, t_data)
            for k, v in losses.items():
                message += '%s: %.3f ' % (k, v)
            print(message)  # print the message

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                with open(opt.resultroot+'train.log','a') as log_file:
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, total_iters, t_comp, t_data)
                    for k, v in losses.items():
                        message += '%s: %.3f ' % (k, v)

                    print(message)  # print the message
                    log_file.write('%s\n' % message)  # save the message

                    model.set_input(dataset[i])  # unpack data from data loader
                    if  opt.model in ['pix2pix']:
                        rA, rB, fB = model.test()           # run inference
                        if opt.save_training_process: 
                            prefix = opt.sample_dir + str(total_iters)
                            save(prefix+'rA.tiff',rA)
                            save(prefix+'rB.tiff',rB)
                            save(prefix+'fB.tiff',fB)
                    elif  opt.model in ['stn']:
                        rA, rB, fB0, fB = model.test()           # run inference
                        if opt.save_training_process: 
                            prefix = opt.sample_dir + str(total_iters)
                            save(prefix+'rA.tiff',rA)
                            save(prefix+'rB.tiff',rB)
                            save(prefix+'fB0.tiff',fB0)
                            save(prefix+'fB.tiff',fB)
                    else:
                        print("Unsupported model. Use pix2pix or stn(for the auto-alignment)")


            if total_iters % opt.save_latest_freq == 0 or total_iters==1000:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        if opt.model in ['stn'] and model.stage in [2,4] : #the previous model stage ==2
            for fnA in adjust_dict:
                fnnA = fnA.split('/')[-1]
                if fnnA not in dataset.stn_adjust_dict:
                    dataset.stn_adjust_dict[fnnA] = [0,0,0]
                adjust_fnA = np.array(adjust_dict[fnA])
                num_data = int(adjust_fnA.shape[0]*1//3)
                adjust_mean_zyx = np.mean(adjust_fnA[num_data:,:],axis=0)
                dataset.stn_adjust_dict[fnnA][0] += adjust_mean_zyx[0]
                dataset.stn_adjust_dict[fnnA][1] += adjust_mean_zyx[1]
                dataset.stn_adjust_dict[fnnA][2] += adjust_mean_zyx[2]
                dataset.stn_adjust_dict[fnnA] = np.array(dataset.stn_adjust_dict[fnnA])
                print(fnnA,adjust_mean_zyx)
                with open(opt.resultroot+'/adjust_mean_zyx.log','a') as fp:
                    fp.write(f'{total_iters},{adjust_mean_zyx[0]},{adjust_mean_zyx[1]},{adjust_mean_zyx[2]}\n')

        odd_epoch_flag = not odd_epoch_flag

if __name__ == '__main__':
    train()