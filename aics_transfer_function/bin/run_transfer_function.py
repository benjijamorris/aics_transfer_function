#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback

# import torch.backends.cudnn as cudnn
import torch

from aics_transfer_function.options import BaseOptions
from aics_transfer_function.proj_trainer import ProjectTrainer
from aics_transfer_function.proj_tester import ProjectTester

# Global object
TRAIN_MODE = "train"
VALID_MODE = "validate"
INFER_MODE = "inference"

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


class Args(argparse.Namespace):
    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(description="runner for training a new model",)
        p.add_argument(
            "--debug", action="store_true", dest="debug", help=argparse.SUPPRESS,
        )
        p.add_argument(
            "--config", dest="filename", help="path to configuration file"
        )
        p.add_argument(
            "--mode", help="the type of operation: train, validation, inference"
        )

        p.parse_args(namespace=self)


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug

        # check gpu option
        assert torch.cuda.is_available(), f"GPU is not available."
        torch.cuda.set_device(torch.device('cuda:0'))

        if args.mode == TRAIN_MODE or args.mode.lower() == TRAIN_MODE:
            opt = BaseOptions(args.filename, isTrain=True).parse()
            exe = ProjectTrainer(opt)
            exe.run_trainer()
        elif args.mode == VALID_MODE or args.mode.lower() == VALID_MODE:
            opt = BaseOptions(args.filename, isTrain=False).parse()
            exe = ProjectTester(opt)
            exe.run_validation()
        elif args.mode == INFER_MODE or args.mode.lower() == INFER_MODE:
            opt = BaseOptions(args.filename, isTrain=False).parse()
            exe = ProjectTester(opt)
            exe.run_inference()
        else:
            log.error(f"Mode {args.mode} is not supported yet")
            sys.exit(1)

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
