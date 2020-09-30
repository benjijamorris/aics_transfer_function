#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import random
import time
from glob import glob
from .dataloader.cyclelarge_dataset import cyclelargeDataset
from .dataloader.cyclelargenopad_dataset import cyclelargenopadDataset
from .models import create_model
from .util.z_align import get_align_kb


class ProjectTrainer(object):
    """
    Main class for training a new transfer function (or continue training)
    """

    def __init__(self, opt):
        """
        Parameters
        ----------
        opt: Dict
            The dictionary of all paramaters/options
        """
        if opt.seed > 0:
            np.random.seed(opt.seed)
            random.seed(opt.seed)
            torch.manual_seed(opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if opt.debug:
            opt.imgs_per_epoch = 3
            opt.patches_per_epoch = 20
            opt.train_num = 5
            print('Debug mode is on. Fewer data will be used.')

        # identify all training data
        filenamesA, filenamesB = self.get_filenames(self.opt.fpath1, self.opt.fpath2)

        if self.opt.train_num > 0:
            filenamesA = filenamesA[:self.opt.train_num]
            filenamesB = filenamesB[:self.opt.train_num]

        if self.opt.imgs_per_epoch > min(len(filenamesA), len(filenamesB)):
            self.opt.imgs_per_epoch = min(len(filenamesA), len(filenamesB))

        self.filenamesA = filenamesA
        self.filenamesB = filenamesB
        self.opt = opt

    @staticmethod
    def get_filenames(p1, p2, s1=None, s2=None, check_name=False):
        all1 = sorted(glob(p1 + '*.tiff') + glob(p1 + '*.tif'))
        all2 = sorted(glob(p2 + '*.tiff') + glob(p2 + '*.tif'))

        assert len(all1) == len(all2), "different number of source and target images"
        for i in range(len(all1)):
            assert os.path.basename(all1[i]) == os.path.basename(all2[i]), \
                f"Filename mismatch: {all1[i]}, {all2[i]}"
        return all1, all2

    def run_trainer(self):
        """
        do the training
        """
        print("running a little setup before full training starts ...")

        # creat dataset class
        if 'nopad' in self.opt.netG:
            dataset = cyclelargenopadDataset(self.opt)
        else:
            dataset = cyclelargeDataset(self.opt)
        self.opt.size_out = dataset.get_size_out()
        self.opt.up_scale = dataset.get_up_scale()

        # create the model
        model = create_model(self.opt)
        model.setup(self.opt)

        print("model setup completes, ready to start training!")

        # start training
        total_iters = 0                # the total number of training iterations
        odd_epoch_flag = True               # T -> F -> T -> F -> T....

        # TODO: check AA code
        adjust_dict = {}  # for stn adjust_dict
        epoch_for_stage_index = 0  # for stn adjust_dict
        first_adjust = True

        epoch_init = self.opt.epoch_count
        for epoch in range(epoch_init, self.opt.niter + self.opt.niter_decay + 1):    
            # outer loop for different epochs; 
            # we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            print('loading/re-loading data begins ...')

            if self.opt.model in ['stn']:
                self.opt.stn_first_stage = str(self.opt.stn_first_stage)
                self.opt.stn_loop_stage = str(self.opt.stn_loop_stage)
                if epoch_for_stage_index >= len(self.opt.stn_first_stage):
                    model.stage = int((self.opt.stn_loop_stage)[((epoch_for_stage_index-len(self.opt.stn_first_stage)))%(len(self.opt.stn_loop_stage))])
                else:
                    model.stage = int((self.opt.stn_first_stage)[epoch_for_stage_index])
                epoch_for_stage_index += 1

                if self.opt.stn_progressive_adjust: # the adjust is based on the previous adjusts. It is a training strategy
                    if model.stage == 2: # the current model stage == 2
                        adjust_dict  =  {}
                else:
                    if self.opt.stn_adjust_image and model.stage != 2:  # the adjusts are independent. It is used to verify the performance of AutoAlign module
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

            if self.opt.sample_mode in ['shift_odd',] and (not ToApplyShift):
                idxA = [minfileA,]
            else:
                idxA = random.sample(range(len(self.filenamesA)), self.opt.imgs_per_epoch)
                idxB = random.sample(range(len(self.filenamesB)), self.opt.imgs_per_epoch)

            if self.opt.model in ['pix2pix', 'stn']:
                idxB = idxA
            fileA = [self.filenamesA[i] for i in idxA]
            if self.opt.name  != 'denoise':
                fileB = [self.filenamesB[i] for i in idxB]
            else:
                fileB = self.filenamesB
            dataset.ToApplyShift = ToApplyShift
            dataset.load_from_file(fileA, fileB, num_patch=self.opt.patches_per_epoch)
            print('reload data finish\n')


            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                if total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
                    model.set_input(dataset[i])  # unpack data from data loader
                    if self.opt.model in ['pix2pix']:
                        rA, rB, fB = model.test()           # run inference
                        if self.opt.save_training_inspections:
                            prefix = self.opt.sample_dir + str(total_iters)
                            save(prefix+'fB_pre.tiff',fB)  # TODO: rewrite save function

                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.self.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if self.opt.model in ['stn'] and self.opt.stn_adjust_image:
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

                if total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    with open(self.opt.resultroot+'train.log','a') as log_file:
                        t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, total_iters, t_comp, t_data)
                        for k, v in losses.items():
                            message += '%s: %.3f ' % (k, v)

                        print(message)  # print the message
                        log_file.write('%s\n' % message)  # save the message

                        model.set_input(dataset[i])  # unpack data from data loader
                        if  self.opt.model in ['pix2pix']:
                            rA, rB, fB = model.test()           # run inference
                            if self.opt.save_training_inspections: 
                                prefix = self.opt.sample_dir + str(total_iters)
                                save(prefix+'rA.tiff',rA)  # TODO: rewrite save function
                                save(prefix+'rB.tiff',rB)  # TODO: rewrite save function
                                save(prefix+'fB.tiff',fB)  # TODO: rewrite save function
                        elif  self.opt.model in ['stn']:
                            rA, rB, fB0, fB = model.test()           # run inference
                            if self.opt.save_training_inspections: 
                                prefix = self.opt.sample_dir + str(total_iters)
                                save(prefix+'rA.tiff',rA)  # TODO: rewrite save function
                                save(prefix+'rB.tiff',rB)  # TODO: rewrite save function
                                save(prefix+'fB0.tiff',fB0)  # TODO: rewrite save function
                                save(prefix+'fB.tiff',fB)  # TODO: rewrite save function
                        else:
                            print("Unsupported model. Use pix2pix or stn(for the auto-alignment)")


                if total_iters % self.opt.save_latest_freq == 0 or total_iters==1000:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
            if epoch % self.opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.niter + self.opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()                     # update learning rates at the end of every epoch.

            if self.opt.model in ['stn'] and model.stage in [2,4] : #the previous model stage ==2
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
                    with open(self.opt.resultroot+'/adjust_mean_zyx.log','a') as fp:
                        fp.write(f'{total_iters},{adjust_mean_zyx[0]},{adjust_mean_zyx[1]},{adjust_mean_zyx[2]}\n')

            odd_epoch_flag = not odd_epoch_flag
