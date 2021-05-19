#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import tifffile

from .models import create_model
from .dataloader.cyclelarge_dataset import cyclelargeDataset
from .util.misc import get_filenames


def extract_filename(filename, replace=False, old_name="", rep_name=""):
    """extract the base name and reuse for output"""

    filename_rev = filename[::-1]
    idx = filename_rev.index("/")
    new = filename_rev[0:idx][::-1]
    if replace:
        new = new.replace(old_name, rep_name)
    return new


def arrange(opt, data, output, position):
    """take care of the sliding window application of the model"""

    data = data[0, 0].cpu().numpy()
    za, ya, xa = position
    patch_size = data.shape

    if za == 0:
        z1 = 0
    else:
        z1 = patch_size[0] // 4

    if za + patch_size[0] == output.shape[0]:
        z2 = patch_size[0]
    else:
        z2 = patch_size[0] // 4 + patch_size[0] // 2

    if ya == 0:
        y1 = 0
    else:
        y1 = patch_size[1] // 4

    if ya + patch_size[1] == output.shape[1]:
        y2 = patch_size[1]
    else:
        y2 = patch_size[1] // 4 + patch_size[1] // 2

    if xa == 0:
        x1 = 0
    else:
        x1 = patch_size[2] // 4

    if xa + patch_size[2] == output.shape[2]:
        x2 = patch_size[2]
    else:
        x2 = patch_size[2] // 4 + patch_size[2] // 2

    zaa = za + z1
    zbb = za + z2
    yaa = ya + y1
    ybb = ya + y2
    xaa = xa + x1
    xbb = xa + x2
    output[zaa:zbb, yaa:ybb, xaa:xbb] = data[z1:z2, y1:y2, x1:x2]


class ProjectTester(object):
    """
    Base class for applying a trained transfer function model for
    validation or inference purpose
    """

    def __init__(self, opt):
        """
        Parameters
        ----------
        opt: Dict
            The dictionary of all paramaters/options
        """

        self.opt = opt
        model = create_model(opt)  # create a model given opt.model and options
        model.setup(opt)  # regular setup: load and print networks
        self.model = model

    def run_inference(self, single_test_image: str = None, normalized: bool = False):
        """
        Main inference function, four different options are supported:
        - use config only, specify one filename in "source"
        - use config only, specify one folder in "source", all ".tif" and ".tiff"
          in the folder will be processed
        - pass in "single_test_image" as a filename
        - pass in "single_test_image" as a numpy array. If "normalized" is True
          the image will be directly processed, otherwise, the image will be
          normalized first using the normalization method in config. Default is
          "normalized" = False
        """

        if single_test_image is not None:
            filenamesA = [
                single_test_image,
            ]
        elif os.path.isfile(self.opt.datapath["source"]):
            filenamesA = [
                self.opt.datapath["source"],
            ]
        else:
            filenamesA = get_filenames(self.opt.datapath["source"])
        dataset = cyclelargeDataset(self.opt, aligned=True)

        self.opt.size_out = dataset.get_size_out()

        for fileA in filenamesA:
            if isinstance(fileA, np.ndarray):
                dataset.load_array(fileA, norm_array=normalized)
            else:
                dataset.load_from_file(
                    [
                        fileA,
                    ]
                )
            position = dataset.positionA
            rA = np.zeros(position[0]).astype("float32")
            fB = np.zeros(position[0]).astype("float32")
            rB = np.zeros(position[0]).astype("float32")

            for i, data in enumerate(dataset):
                self.model.set_input(data)  # unpack data from data loader

                rA_i, rB_i, fB_i = self.model.test()
                arrange(self.opt, rA_i, rA, position[i + 1])
                arrange(self.opt, rB_i, rB, position[i + 1])
                arrange(self.opt, fB_i, fB, position[i + 1])

            if single_test_image is not None:
                return fB
            else:
                ########################################################################
                # Temp saving script
                filename_ori = extract_filename(
                    fileA, replace=True, old_name="source.tif", rep_name="pred.tiff"
                )
                tif = tifffile.TiffWriter(
                    self.opt.output_path / filename_ori, bigtiff=True
                )
                tif.save(fB, compress=9, photometric="minisblack", metadata=None)
                tif.close()
                print(filename_ori + " saved")
                ########################################################################

    def run_validation(self):

        from skimage.metrics import structural_similarity as ssim
        import pandas as pd

        filenamesA, filenamesB = get_filenames(
            self.opt.datapath["source"], self.opt.datapath["target"]
        )
        dataset = cyclelargeDataset(self.opt, aligned=True)

        self.opt.size_out = dataset.get_size_out()

        val_report = []

        for fileA, fileB in zip(filenamesA, filenamesB):
            dataset.load_from_file(
                [
                    fileA,
                ],
                [
                    fileB,
                ],
            )
            position = dataset.positionB
            positionA = dataset.positionA
            rA = np.zeros(positionA[0]).astype("float32")
            rB = np.zeros(position[0]).astype("float32")
            fB = np.zeros(position[0]).astype("float32")

            for i, data in enumerate(dataset):
                self.model.set_input(data)  # unpack data from data loader

                rA_i, rB_i, fB_i = self.model.test()
                arrange(self.opt, rA_i, rA, positionA[i + 1])
                arrange(self.opt, rB_i, rB, position[i + 1])
                arrange(self.opt, fB_i, fB, position[i + 1])

            ###########################################################################
            # Temp saving script
            filename_ori = extract_filename(
                fileA, replace=True, old_name="source.tif", rep_name="pred.tiff"
            )
            tif = tifffile.TiffWriter(self.opt.output_path / filename_ori, bigtiff=True)
            tif.save(fB, compress=9, photometric="minisblack", metadata=None)
            tif.close()
            print(filename_ori + " saved")
            ###########################################################################

            # calculate pearson correliation and ssim
            corr_mat = np.corrcoef(np.ravel(rB), np.ravel(fB))
            corr_value = corr_mat[1, 0]

            ssim_value = ssim(rB, fB, data_range=fB.max() - fB.min())

            val_report.append(
                {"name": filename_ori, "corr": corr_value, "ssim": ssim_value}
            )

        val_df = pd.DataFrame(val_report)
        val_df.to_csv(self.opt.output_path / "val_report.csv")
