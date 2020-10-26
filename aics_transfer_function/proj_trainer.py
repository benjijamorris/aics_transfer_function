#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import torch
import random
import time
from .dataloader.cyclelarge_dataset import cyclelargeDataset
from .models import create_model
from .util.misc import save_tensor as save
from .util.misc import get_filenames


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
            opt.training_setting["imgs_per_epoch"] = 3
            opt.training_setting["patches_per_epoch"] = 20
            opt.training_setting["train_num"] = 5
            print("Debug mode is on. Fewer data will be used.")

        # identify all training data
        filenamesA, filenamesB = get_filenames(
            opt.datapath["source"], opt.datapath["target"]
        )

        # keep only a portion of the training data
        if opt.training_setting["train_num"] > 0:
            filenamesA = filenamesA[: opt.training_setting["train_num"]]
            filenamesB = filenamesB[: opt.training_setting["train_num"]]

        if (
            opt.training_setting["imgs_per_epoch"]
            > min(len(filenamesA), len(filenamesB))
            or opt.training_setting["imgs_per_epoch"] < 0
        ):

            opt.training_setting["imgs_per_epoch"] = min(
                len(filenamesA), len(filenamesB)
            )

        self.filenamesA = filenamesA
        self.filenamesB = filenamesB
        self.opt = opt

    def run_trainer(self):
        """
        do the training
        """
        print("running a little setup before full training starts ...")
        # creat dataset class
        dataset = cyclelargeDataset(self.opt)
        self.opt.size_out = dataset.get_size_out()
        self.opt.up_scale = dataset.get_up_scale()

        # create the model
        model = create_model(self.opt)
        model.setup(self.opt)

        print("model setup completes, ready to start training!")

        # start training
        total_iters = 0  # the total number of training iterations
        odd_epoch_flag = True  # T -> F -> T -> F -> T....

        # TODO: check AA code
        adjust_dict = {}  # for stn adjust_dict
        epoch_for_stage_index = 0  # for stn adjust_dict
        total_epoch = (
            self.opt.training_setting["niter"]
            + self.opt.training_setting["niter_decay"]
            + 1
        )
        for epoch in range(1, total_epoch):
            # outer loop for different epochs;
            # we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch

            # the number of training iterations in current epoch, reset to 0 every epoch
            epoch_iter = 0

            print("loading/re-loading data begins ...")

            # TODO: check AA code
            if self.opt.network["model"] in ["stn"]:
                self.opt.stn_first_stage = str(self.opt.stn_first_stage)
                self.opt.stn_loop_stage = str(self.opt.stn_loop_stage)
                if epoch_for_stage_index >= len(self.opt.stn_first_stage):
                    s_id = ((epoch_for_stage_index - len(self.opt.stn_first_stage))) % (
                        len(self.opt.stn_loop_stage)
                    )
                    model.stage = int((self.opt.stn_loop_stage)[s_id])
                else:
                    model.stage = int((self.opt.stn_first_stage)[epoch_for_stage_index])
                epoch_for_stage_index += 1

                # the adjust is based on the previous adjusts. It is a training strategy
                if self.opt.stn_progressive_adjust:
                    if model.stage == 2:  # the current model stage == 2
                        adjust_dict = {}
                else:
                    # the adjusts are independent. It is used to verify the performance
                    # of AutoAlign module
                    if self.opt.stn_adjust_image and model.stage != 2:
                        for fnA in adjust_dict:
                            fnnA = fnA.split("/")[-1]
                            adjust_fnA = np.array(adjust_dict[fnA])
                            num_data = int(adjust_fnA.shape[0] * 1 // 3)
                            adjust_mean_zyx = np.mean(adjust_fnA[num_data:, :], axis=0)
                            dataset.stn_adjust_dict[fnnA] = adjust_mean_zyx
                            print(fnnA, adjust_mean_zyx)
                    else:
                        # adjust_dict: {filenameA: [shift1, shift2, ...]; filenameB ...}
                        # dataset.stn_adjust_dict: {filenameA: (z,y,x);filenameB: ...}
                        dataset.stn_adjust_dict = {}
                        adjust_dict = {}

            idxA = random.sample(
                range(len(self.filenamesA)), self.opt.training_setting["imgs_per_epoch"]
            )

            if self.opt.network["model"] in ["pix2pix", "stn"]:
                idxB = idxA

            fileA = [self.filenamesA[i] for i in idxA]
            fileB = [self.filenamesB[i] for i in idxB]
            dataset.load_from_file(
                fileA, fileB, num_patch=self.opt.training_setting["patches_per_epoch"]
            )
            print("loading/reloading is done \n")

            # inner loop within one epoch
            for i, data in enumerate(dataset):
                iter_start_time = time.time()  # timer per iteration

                # unpack data from dataset and apply preprocessing
                model.set_input(data)

                # calculate loss functions, get gradients, update network weights
                model.optimize_parameters()

                # TODO: check AA code
                if self.opt.network["model"] in ["stn"]:
                    if self.opt.model["stn_adjust_image"] and model.stage == 2:
                        fnA = data["A_paths"]
                        shift_zyx = model.get_shift().cpu().detach().numpy()
                        if fnA in adjust_dict:
                            adjust_dict[fnA].append(shift_zyx)
                        else:
                            adjust_dict[fnA] = [
                                shift_zyx,
                            ]

                # just a quick sanity check
                if total_iters == 1:
                    print(data["A"].shape)
                    print(data["B"].shape)

                # print the loss
                losses = model.get_current_losses()
                message = f"(epoch: {epoch}, iters: {total_iters})"
                for k, v in losses.items():
                    message += f"{k}: {v}"
                print(message)

                if total_iters % self.opt.save["print_freq"] == 0:
                    losses = model.get_current_losses()
                    with open(self.opt.resultroot / Path("train.log"), "a") as log_file:
                        tc = (
                            time.time() - iter_start_time
                        ) / self.opt.training_setting["batch_size"]
                        message = f"(epoch: {epoch}, iters: {total_iters}, time: {tc})"
                        for k, v in losses.items():
                            message += f"{k}: {v}"

                        print(message)
                        log_file.write("%s\n" % message)  # save the message

                        model.set_input(dataset[i])  # unpack data from data loader
                        if self.opt.network["model"] in ["pix2pix"]:
                            rA, rB, fB = model.test()
                            if self.opt.save["save_training_inspections"]:
                                save(
                                    self.opt.sample_dir
                                    / Path(str(total_iters) + "rA.tiff"),
                                    rA,
                                )
                                save(
                                    self.opt.sample_dir
                                    / Path(str(total_iters) + "rB.tiff"),
                                    rB,
                                )
                                save(
                                    self.opt.sample_dir
                                    / Path(str(total_iters) + "fB.tiff"),
                                    fB,
                                )
                        elif self.opt.network["model"] in ["stn"]:
                            rA, rB, fB0, fB = model.test()
                            if self.opt.save["save_training_inspections"]:
                                save(
                                    self.opt.sample_dir
                                    / Path(str(total_iters) + "rA.tiff"),
                                    rA,
                                )
                                save(
                                    self.opt.sample_dir
                                    / Path(str(total_iters) + "rB.tiff"),
                                    rB,
                                )
                                save(
                                    self.opt.sample_dir
                                    / Path(str(total_iters) + "fB0.tiff"),
                                    fB0,
                                )
                                save(
                                    self.opt.sample_dir
                                    / Path(str(total_iters) + "fB.tiff"),
                                    fB,
                                )

                if (
                    total_iters % self.opt.save["save_latest_freq"] == 0
                    or total_iters == 100
                ):
                    print(f"save latest model (epoch {epoch}, iters {total_iters})")
                    model.save_networks("latest")

                total_iters += self.opt.training_setting["batch_size"]
                epoch_iter += self.opt.training_setting["batch_size"]

            if epoch % self.opt.save["save_epoch_freq"] == 0:
                print(f"saving the model at the end of epoch {epoch}")
                model.save_networks("latest")
                model.save_networks(epoch)

            tt = (
                self.opt.training_setting["niter"]
                + self.opt.training_setting["niter_decay"]
            )
            print(f"End of epoch {epoch} / {tt}")
            print(f"Time Taken: {time.time() - epoch_start_time} sec")

            # update learning rates at the end of every epoch.
            model.update_learning_rate()

            # TODO: check AA code
            # the previous model stage == 2
            if self.opt.network["model"] == "stn" and model.stage in [2, 4]:
                for fnA in adjust_dict:
                    fnnA = fnA.split("/")[-1]
                    if fnnA not in dataset.stn_adjust_dict:
                        dataset.stn_adjust_dict[fnnA] = [0, 0, 0]
                    adjust_fnA = np.array(adjust_dict[fnA])
                    num_data = int(adjust_fnA.shape[0] * 1 // 3)
                    adjust_mean_zyx = np.mean(adjust_fnA[num_data:, :], axis=0)
                    dataset.stn_adjust_dict[fnnA][0] += adjust_mean_zyx[0]
                    dataset.stn_adjust_dict[fnnA][1] += adjust_mean_zyx[1]
                    dataset.stn_adjust_dict[fnnA][2] += adjust_mean_zyx[2]
                    dataset.stn_adjust_dict[fnnA] = np.array(
                        dataset.stn_adjust_dict[fnnA]
                    )
                    print(fnnA, adjust_mean_zyx)
                    adj_log_name = self.opt.resultroot + "/adjust_mean_zyx.log"
                    with open(adj_log_name, "a") as fp:
                        fp.write(
                            f"{total_iters},{adjust_mean_zyx[0]},"
                            f"{adjust_mean_zyx[1]},{adjust_mean_zyx[2]}\n"
                        )

            odd_epoch_flag = not odd_epoch_flag
