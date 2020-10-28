import numpy as np
import os
import importlib
import random
from collections import OrderedDict
import torch
from torch import from_numpy
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from aicsimageio import AICSImage
from aicsimageprocessing import resize_to
from tqdm import tqdm


class cyclelargeDataset(Dataset):
    """
    Dataloader for reading the training data and returns dataset class
    """

    def __init__(self, opt, aligned=True):
        Dataset.__init__(self)
        self.imgA = []
        self.imgB = []
        self.size_in = opt.network["input_patch_size"]
        self.opt = opt
        self.filenamesA = None
        self.filenamesB = None
        if opt.isTrain:  # only need a name when training
            self.name = opt.name
        self.batch_size = opt.network["batch_size"]
        self.resizeA = opt.resizeA
        self.netG = opt.network["netG"]
        self.model = opt.network["model"]
        self.shift_dict = {}

        # TODO: check AA code
        # similar to shift_dict, only for stn opt.stn_adjust_image use
        self.stn_adjust_dict = {}

        self.up_scale = (1, 1, 1)
        self.size_out = self.size_in
        self.aligned = aligned

        module_name = "aics_transfer_function.util.preprocessing"
        norm_module = importlib.import_module(module_name)
        func_name_src = self.opt["normalization"]["source"]["method"]
        self.source_norm = getattr(norm_module, func_name_src)
        self.source_norm_param = self.opt["normalization"]["source"]["params"]
        if "target" in self.opt.datapath and self.opt.datapath["target"] is not None:
            func_name_tar = self.opt["normalization"]["target"]["method"]
            self.target_norm = getattr(norm_module, func_name_tar)
            self.target_norm_param = self.opt["normalization"]["target"]["params"]

    def load_array(self, src_img, norm_array: bool = False):
        self.imgA = []
        self.imgB = []
        self.imgA_path = []
        self.imgA_short_path = []
        self.imgB_path = []

        self.filenamesA = ["array"]

        # TODO: check AA code
        # True if this patch is to be used for calculating mean offset for AutoAlign
        self.for_calc_ave_offset = []

        if not norm_array:
            src_img = self.source_norm(src_img, bulk_params=self.source_norm_param)
            r = self.opt.normalization["source"]["ratio_param"]
            nz = int(np.round(src_img.shape[0] * r[0]))
            ny = int(np.round(src_img.shape[1] * r[1]))
            nx = int(np.round(src_img.shape[2] * r[2]))
            new_size = (nz, ny, nx)
            src_img = resize_to(src_img, new_size, method="bilinear")
        else:
            new_size = src_img.shape

        overlap_step = 0.5
        self.positionA = [
            new_size,
        ]
        px_list, py_list, pz_list = [], [], []
        px, py, pz = 0, 0, 0
        while px < new_size[2] - self.size_in[2]:
            px_list.append(px)
            px += int(self.size_in[2] * overlap_step)
        px_list.append(new_size[2] - self.size_in[2])
        while py < new_size[1] - self.size_in[1]:
            py_list.append(py)
            py += int(self.size_in[1] * overlap_step)
        py_list.append(new_size[1] - self.size_in[1])
        while pz < new_size[0] - self.size_in[0]:
            pz_list.append(pz)
            pz += int(self.size_in[0] * overlap_step)
        pz_list.append(new_size[0] - self.size_in[0])
        for pz_in in pz_list:
            for py_in in py_list:
                for px_in in px_list:
                    (self.imgA).append(
                        np.expand_dims(
                            src_img[
                                pz_in : pz_in + self.size_in[0],
                                py_in : py_in + self.size_in[1],
                                px_in : px_in + self.size_in[2],
                            ],
                            axis=0,
                        )
                    )
                    (self.imgA_path).append("default")
                    (self.imgA_short_path).append("default_short")

                    self.positionA.append((pz_in, py_in, px_in))
                    self.for_calc_ave_offset.append(False)

    def load_from_file(self, filenamesA, filenamesB=None, num_patch=-1):
        # assumption: transfer is from A to B
        self.imgA = []
        self.imgB = []
        self.imgA_path = []
        self.imgA_short_path = []
        self.imgB_path = []

        if filenamesB is not None:
            assert len(filenamesA) == len(filenamesB), "source/target num mismatch"

        num_data = len(filenamesA)
        assert num_data > 0, "no source type data found"

        # how many patches to take from each image
        self.num_patch_per_img = np.zeros((num_data,), dtype=int)

        if num_patch == -1:
            # take all patches in a "stitch" way
            Stitch = True
        else:
            Stitch = False
            if num_data >= num_patch:
                print("suggest to use more patch in each buffer")
                self.num_patch_per_img[:num_patch] = 1
            else:
                basic_num = num_patch // num_data
                self.num_patch_per_img[:] = basic_num
                self.num_patch_per_img[: (num_patch - basic_num * num_data)] = (
                    self.num_patch_per_img[: (num_patch - basic_num * num_data)] + 1
                )

        # TODO: to be cleaned, this is reserved for cycle gan
        if (not self.aligned) and (not Stitch):
            shuffle(filenamesA)
            shuffle(filenamesB)
        self.filenamesA = filenamesA
        self.filenamesB = filenamesB

        # TODO: check AA code
        # True if this patch is to be used for calculating mean offset for AutoAlign
        self.for_calc_ave_offset = []

        if self.opt.network == "stn" and self.opt.network["stn_adjust_fixed_z"]:
            print(f"read offsets from {self.opt.readoffsetfrom}")
            assert os.path.isfile(
                self.opt.readoffsetfrom
            ), f"opt.readoffsetfrom path: {self.opt.readoffsetfrom} is not found! \
                If you want to align images, set the correct path. If not, \
                set opt.stn_adjust_fixed_z=False"
            with open(self.opt.readoffsetfrom, "r") as fp:
                fixed_dict1 = {}
                for line in fp.readlines():
                    key, z, y, x = line.strip().split(",")
                    z = float(z)
                    y = float(y)
                    x = float(x)
                    if key in fixed_dict1:
                        fixed_dict1[key].append([z, y, x])
                    else:
                        fixed_dict1[key] = [
                            [z, y, x],
                        ]
                for key in fixed_dict1:
                    # only use the parameters that are between [10,90] percents.
                    clip_param = np.percentile(
                        np.array(fixed_dict1[key])[:, 0], [10, 90]
                    )
                    offset_raw = []
                    for i in range(len(fixed_dict1[key])):
                        if (
                            fixed_dict1[key][i][0] > clip_param[0]
                            and fixed_dict1[key][i][0] < clip_param[1]
                        ):
                            offset_raw.append(fixed_dict1[key][i])
                    offset_raw = np.array(offset_raw)
                    z_std = np.std(offset_raw, axis=0)[0]

                    if z_std > 0.5:
                        print(
                            f"WARNING: The standard deviation of offsets estimation\
                             for {key} is {z_std}. Not accurate!"
                        )
                        with open(self.opt.resultroot + "WARNING", "w") as wp:
                            wp.write(
                                f"WARNING: The standard deviation of offsets \
                                estimation for {key} is {z_std}. Not accurate!"
                            )
                    fixed_dict1[key] = np.mean(offset_raw, axis=0)

        for idxA, fnA in tqdm(enumerate(filenamesA)):
            fnnA = fnA.split("/")[-1]

            # expected patch is met (when num_patch = -1, the loading will go thr all)
            if len(self.imgA) == num_patch:
                break

            # load source domain image
            source_reader = AICSImage(fnA)  # STCZYX
            src_img = source_reader.get_image_data("ZYX", S=0, T=0, C=0)

            # run intensity normalization
            src_img = self.source_norm(src_img, bulk_params=self.source_norm_param)

            if filenamesB is not None:
                idxB = idxA
                fnB = filenamesB[idxB]

                # load target domain image
                target_reader = AICSImage(fnB)  # STCZYX
                tar_img = target_reader.get_image_data("ZYX", S=0, T=0, C=0)

                # run intensity normalization
                tar_img = self.target_norm(tar_img, bulk_params=self.target_norm_param)

                # determine new size for source
                new_size = (tar_img.shape[0], tar_img.shape[1], tar_img.shape[2])

            else:
                r = self.opt.normalization["source"]["ratio_param"]
                nz = int(np.round(src_img.shape[0] * r[0]))
                ny = int(np.round(src_img.shape[1] * r[1]))
                nx = int(np.round(src_img.shape[2] * r[2]))
                new_size = (nz, ny, nx)

            src_img = resize_to(src_img, new_size, method="bilinear")

            # TODO: check AA code
            """
            if (self.opt.network["model"] in ['stn'] and self.opt.stn_adjust_image):
                if self.opt.isTrain:
                    shifted_stacks_dir = self.opt.resultroot + '/shift/'
                    if not os.path.isdir(shifted_stacks_dir):
                        os.makedirs(shifted_stacks_dir)
                    if fnnA in self.stn_adjust_dict:
                        imsave(shifted_stacks_dir + f'{fnnA}_rA.tiff', src_img[0])
                        print(fnnA, self.stn_adjust_dict[fnnA])
                        offsets_zyx = self.stn_adjust_dict[fnnA]
                        offsets_zyx[0] = offsets_zyx[0] * 1.0 / self.up_scale[0]
                        offsets_zyx[1] = offsets_zyx[1] * 1.0 / self.up_scale[1]
                        offsets_zyx[2] = offsets_zyx[2] * 1.0 / self.up_scale[2]
                        with open(shifted_stacks_dir + 'shift.log', 'a') as fp:
                            fp.write(f"{fnnA},{offsets_zyx[0]},{offsets_zyx[1]},\
                                {offsets_zyx[2]}\n")
                        offsets_zyx = from_numpy(offsets_zyx)
                        tensor = from_numpy(np.expand_dims(src_img, 0))
                        label = self.apply_adjust(tensor, offsets_zyx)
                        label = np.squeeze(label.detach().cpu().numpy(), axis=0)
                        imsave(shifted_stacks_dir + f'{fnnA}_rA_new.tiff', src_img[0])
            elif self.opt.stn_adjust_fixed_z:
                if self.opt.isTrain:
                    print(f'adjust fixed z')
                    if fnnA in fixed_dict1:
                        z, y, x = fixed_dict1[fnnA]
                    else:
                        z, y, x = 0, 0, 0
                        raise ValueError(f"****\n\nERROR: {fnnA} is not found \
                            in {self.opt.readoffsetfrom}, please train the AutoAlign \
                                first to get the offsets\n\n ***************\n")
                    # z = fixed_dict1_deprecated[int(fnnA[:3])]
                    if self.opt.align_all_axis:
                        offsets_zyx = np.array((z / self.up_scale[0],
                                                y / self.up_scale[1],
                                                x / self.up_scale[2]))
                    else:
                        offsets_zyx = np.array((z / self.up_scale[0], 0, 0))
                    offsets_zyx = from_numpy(offsets_zyx)
                    tensor = from_numpy(np.expand_dims(src_img, 0))
                    label = self.apply_adjust(tensor, offsets_zyx)
                    label = np.squeeze(label.detach().cpu().numpy(), axis=0)
                    label = label[:, 1:-1, :, :]
                    tar_img = tar_img[
                        :, int(self.up_scale[0]):-int(self.up_scale[0]),
                        :, :
                    ]
            """

            if Stitch:
                overlap_step = 0.5
                if filenamesB is not None:
                    self.positionB = [
                        new_size,
                    ]
                self.positionA = [
                    new_size,
                ]
                px_list, py_list, pz_list = [], [], []
                px, py, pz = 0, 0, 0
                while px < new_size[2] - self.size_in[2]:
                    px_list.append(px)
                    px += int(self.size_in[2] * overlap_step)
                px_list.append(new_size[2] - self.size_in[2])
                while py < new_size[1] - self.size_in[1]:
                    py_list.append(py)
                    py += int(self.size_in[1] * overlap_step)
                py_list.append(new_size[1] - self.size_in[1])
                while pz < new_size[0] - self.size_in[0]:
                    pz_list.append(pz)
                    pz += int(self.size_in[0] * overlap_step)
                pz_list.append(new_size[0] - self.size_in[0])
                for pz_in in pz_list:
                    for py_in in py_list:
                        for px_in in px_list:
                            (self.imgA).append(
                                np.expand_dims(
                                    src_img[
                                        pz_in : pz_in + self.size_in[0],
                                        py_in : py_in + self.size_in[1],
                                        px_in : px_in + self.size_in[2],
                                    ],
                                    axis=0,
                                )
                            )
                            (self.imgA_path).append(fnA)
                            (self.imgA_short_path).append(fnnA)
                            pz_out = pz_in * self.up_scale[0]
                            py_out = py_in * self.up_scale[1]
                            px_out = px_in * self.up_scale[2]

                            if filenamesB is not None:
                                (self.imgB).append(
                                    np.expand_dims(
                                        tar_img[
                                            pz_out : pz_out + self.size_out[0],
                                            py_out : py_out + self.size_out[1],
                                            px_out : px_out + self.size_out[2],
                                        ],
                                        axis=0,
                                    )
                                )
                                (self.imgB_path).append(fnB)
                                self.positionB.append((pz_out, py_out, px_out))

                            self.positionA.append((pz_in, py_in, px_in))
                            self.for_calc_ave_offset.append(False)
            else:
                # TODO: data augmentation, only cropping now
                new_patch_num = 0
                while new_patch_num < self.num_patch_per_img[idxA]:
                    pz = random.randint(0, tar_img.shape[0] - self.size_in[0])
                    py = random.randint(0, tar_img.shape[1] - self.size_in[1])
                    px = random.randint(0, tar_img.shape[2] - self.size_in[2])
                    (self.imgA).append(
                        np.expand_dims(
                            src_img[
                                pz : pz + self.size_in[0],
                                py : py + self.size_in[1],
                                px : px + self.size_in[2],
                            ],
                            axis=0,
                        )
                    )
                    (self.imgA_path).append(fnA)
                    (self.imgA_short_path).append(fnnA)

                    # TODO: good crop?
                    if not self.aligned:
                        pz = random.randint(0, tar_img.shape[1] - self.size_out[0])
                        py = random.randint(0, tar_img.shape[2] - self.size_out[1])
                        px = random.randint(0, tar_img.shape[3] - self.size_out[2])
                    else:
                        pz = pz * self.up_scale[0]
                        py = py * self.up_scale[1]
                        px = px * self.up_scale[2]
                    (self.imgB).append(
                        np.expand_dims(
                            tar_img[
                                pz : pz + self.size_out[0],
                                py : py + self.size_out[1],
                                px : px + self.size_out[2],
                            ],
                            axis=0,
                        )
                    )
                    (self.imgB_path).append(fnB)
                    if new_patch_num > self.num_patch_per_img[idxA] * 1 // 3:
                        self.for_calc_ave_offset.append(True)
                    else:
                        self.for_calc_ave_offset.append(False)
                    new_patch_num += 1

    def apply_shift(self, img1, img2, z):
        z = int(np.round(z))
        assert len(img1.shape) == 4
        assert len(img2.shape) == 4
        if z == 0:
            return img1, img2
        elif z > 0:
            new_img1 = img1[:, :-z, :, :]
            new_img2 = img2[:, z:, :, :]
        else:  # z<0
            new_img1 = img1[:, -z:, :, :]
            new_img2 = img2[:, :z, :, :]
        return new_img1, new_img2

    def apply_adjust(self, tensor, offsets_zyx):
        """
        dz>0: stack goes up (-z:-1)=padding
        dy>0: goes up; dx>0: goes right
        """
        assert len(tensor.shape) == 5
        self.device = torch.device("cuda:{}".format(self.opt.gpu_ids[0]))
        tensor = tensor.to(self.device)
        dz, dy, dx = offsets_zyx.type(torch.float32)
        nz, ny, nx = tensor.shape[2:]
        z_linspace = torch.linspace(-1, 1, steps=nz, device=self.device) + dz * 2.0 / nz
        y_linspace = torch.linspace(-1, 1, steps=ny, device=self.device) + dy * 2.0 / ny
        x_linspace = torch.linspace(-1, 1, steps=nx, device=self.device) + dx * 2.0 / nx
        z_grid, y_grid, x_grid = torch.meshgrid(z_linspace, y_linspace, x_linspace)
        grid = torch.cat(
            [x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), z_grid.unsqueeze(-1)], dim=-1
        ).unsqueeze(0)
        return F.grid_sample(tensor, grid, padding_mode="border")

    def get_fdict(self, filenamesA, filenamesB):
        fdict = OrderedDict()
        fnA_dir = os.path.dirname(filenamesA[0])
        for fnB in filenamesB:
            fnB_base = os.path.basename(fnB)
            fnA = fnA_dir + "/" + fnB_base[:7] + "_ground_truth.tiff"
            if fnA not in filenamesA:
                continue
            if fnA not in fdict:
                fdict[fnA] = [
                    fnB,
                ]
            else:
                fdict[fnA].append(fnB)
        print("Dict is built. Keys:")
        for key in fdict:
            print(key)
        return fdict

    def __getitem__(self, index):
        # TODO: use dataloader
        assert self.filenamesA is not None, "please load_from_file first"
        idxA = index * self.batch_size
        if idxA + self.batch_size > len(self.imgA):
            raise IndexError("end of one epoch")
        image_tensorA = from_numpy(
            np.array(self.imgA[idxA : idxA + self.batch_size]).astype(float)
        )

        if self.filenamesB is not None:
            if self.aligned:
                idxB = np.array(range(idxA, idxA + self.batch_size)).tolist()
            else:
                idxB = np.random.randint(0, len(self.imgB), self.batch_size).tolist()
            image_tensorB = from_numpy(
                np.array([self.imgB[idx] for idx in idxB]).astype(float)
            )
            pathB = [self.imgB_path[idx] for idx in idxB][0]
        else:
            # HACK: need better implementation
            image_tensorB = image_tensorA
            pathB = self.imgA_path[idxA]

        return {
            "A": image_tensorA.float(),
            "B": image_tensorB.float(),
            "A_paths": self.imgA_path[idxA],
            "B_paths": pathB,
            "calcAveOffset": self.for_calc_ave_offset[idxA],
            "A_short_path": self.imgA_short_path[idxA],
        }  # not support batch_size > 1

    def __len__(self):
        return len(self.imgA)

    def get_size_out(self):
        return self.size_out

    def get_up_scale(self):
        return self.up_scale

    def get_num_patch_per_img(self):
        return self.num_patch_per_img
