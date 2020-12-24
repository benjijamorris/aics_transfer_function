import sys
import os
import logging
import argparse
import traceback
from glob import glob
from tqdm import tqdm
import statistics
import numpy as np
import random
from aicsimageio import imread
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu


log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)


def suggest_normalization_param(img):

    ##########################################################
    # take middle chunk
    img = np.squeeze(img)
    img_smooth = gaussian_filter(
        img.astype(np.float32), sigma=1.0, mode="nearest", truncate=3.0
    )
    th = threshold_otsu(img_smooth)
    img_bw = img_smooth > th
    low_chunk = 0
    high_chunk = img.shape[0]
    for zz in range(img.shape[0]):
        if np.count_nonzero(img_bw[zz, :, :] > 0) > 50:
            if zz > 0:
                low_chunk = zz - 1
            break

    for zz in range(img.shape[0]):
        if np.count_nonzero(img_bw[img.shape[0] - zz - 1, :, :] > 0) > 50:
            if zz > 0:
                high_chunk = img.shape[0] - zz
            break

    structure_img0 = img[low_chunk:high_chunk, :, :]
    ##########################################################

    m = np.mean(structure_img0)
    s = np.std(structure_img0)
    # m, s = norm.fit(structure_img0.ravel())

    p99 = np.percentile(structure_img0, 99.99)
    p01 = np.percentile(structure_img0, 0.01)

    pmin = structure_img0.min()
    pmax = structure_img0.max()

    up_ratio = 0
    for up_i in np.arange(0.5, 1000, 0.5):
        if m + s * up_i > p99:
            if m + s * up_i > pmax:
                up_ratio = up_i - 0.5
            else:
                up_ratio = up_i
            break

    low_ratio = 0
    for low_i in np.arange(0.5, 1000, 0.5):
        if m - s * low_i < p01:
            if m - s * low_i < pmin and low_ratio > 0.5:
                low_ratio = low_i - 0.5
            else:
                low_ratio = low_i
            break

    return low_ratio, up_ratio


def preproc(ds_path, ratio):

    # get filenames
    filenames = glob(ds_path + "/*.tiff") + glob(ds_path + "/*.tif")
    random.shuffle(filenames)

    if ratio < 1:
        filenames = filenames[: int(round(ratio * len(filenames)))]

    set_a = []
    set_b = []
    for fn in tqdm(filenames):
        img = np.squeeze(imread(fn))
        a, b = suggest_normalization_param(img)
        set_a.append(a)
        set_b.append(b)

    print(set_a)
    print(set_b)

    print(statistics.mean(set_a))
    print(statistics.mean(set_b))

    print(statistics.stdev(set_a))
    print(statistics.stdev(set_b))

    print(statistics.mean(set_a) + 3 * statistics.stdev(set_a))
    print(statistics.mean(set_b) + 3 * statistics.stdev(set_b))

    print("please use")

    print(max(set_a))
    print(max(set_b))


###############################################################################

###############################################################################


class Args(object):
    """
    Use this to define command line arguments and use them later.

    For each argument do the following
    1. Create a member in __init__ before the self.__parse call.
    2. Provide a default value here.
    3. Then in p.add_argument, set the dest parameter to that variable name.

    See the debug parameter as an example.
    """

    def __init__(self, log_cmdline=True):

        # self.debug = False
        # self.output_dir = './'
        # self.struct_ch = 0
        # self.xy = 0.108

        #
        self.__parse()
        #
        if self.debug:
            log.setLevel(logging.DEBUG)
            log.debug("-" * 80)
            self.show_info()
            log.debug("-" * 80)

    @staticmethod
    def __no_args_print_help(parser):
        """
        This is used to print out the help if no arguments are provided.
        """
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(1)

    def __parse(self):
        p = argparse.ArgumentParser()
        # Add arguments
        p.add_argument(
            "--debug",
            "-d",
            action="store_true",
            dest="debug",
            help="If set debug log output is enabled",
        )
        p.add_argument(
            "--ratio",
            default=1.0,
            type=float,
            help="the name of the structure to be processed",
        )
        p.add_argument("--source_domain", required=True, help="source")
        p.add_argument("--target_domain", required=True, help="target")

        self.__no_args_print_help(p)
        p.parse_args(namespace=self)

    def show_info(self):
        log.debug("Working Dir:")
        log.debug("\t{}".format(os.getcwd()))
        log.debug("Command Line:")
        log.debug("\t{}".format(" ".join(sys.argv)))
        log.debug("Args:")
        for (k, v) in self.__dict__.items():
            log.debug("\t{}: {}".format(k, v))


###############################################################################


class Executor(object):
    def __init__(self, args):

        pass

    def execute(self, args):

        # "/allen/aics/assay-dev/computational/data/transfer_function_feasibility/h2b_data/h2b_20x_100x/alignment_low_snr/complete"

        ###############################################
        # process source domain
        ###############################################
        print("starting source")
        preproc(args.source_domain, args.ratio)

        ###############################################
        # process target domain
        ###############################################
        print("starting target")
        preproc(args.target_domain, args.ratio)


##################################################################################


def main():
    dbg = False
    try:
        args = Args()
        dbg = args.debug

        # Do your work here - preferably in a class or function,
        # passing in your args. E.g.
        exe = Executor(args)
        exe.execute(args)

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
