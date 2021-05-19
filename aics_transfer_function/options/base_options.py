import os
from pathlib import Path
from typing import Union
import shutil
import yaml
import datetime
import git
from munch import Munch

from aics_transfer_function.util.quilt_utils import QuiltModelZoo


class BaseOptions:
    """
    This class defines options used during both training and test time.
    """

    def __init__(self, config_file, running_mode):
        self.running_mode = running_mode
        self.config_file = config_file

    def get_job_name(self, name: str = "TF"):
        """
        Generate a unique name for this training experiment

        Parameters
        -----------
        name: str
            the name of your experiment, e.g. 20xto100x, 100xtoSR, etc.

        Returns
        -----------
        fullname: str
            the full job name in the format of
            jobname_gitSHA_time
        """

        now = datetime.datetime.now()
        time = now.strftime("%m%d_%H%M")
        try:
            repo = git.Repo(search_parent_directories=True)
            gitsha = (repo.head.object.hexsha)[-4:]
        except Exception:
            gitsha = "NoGitSHA"

        fullname = f"{name}_{gitsha}_{time}"
        return fullname

    def print_options(self, opt):
        """
        Print options
        """
        message = "\n----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            message += f"{str(k)}: {str(v)}\n"
        message += "\n----------------- End -------------------\n"
        print(message)

    def parse(self, local_dir: Union[str, Path] = "./"):
        """
        parse the option arguments and fill with default values when missing
        """

        # check if the configuration is local or needs to be pulled from quilt
        if not os.path.exists(self.config_file):
            # pull from quilt
            config_path = Path(local_dir) / f"{self.config_file}.yaml"
            zoo_client = QuiltModelZoo()
            zoo_client.download_model(self.config_file, config_path)
        else:
            config_path = self.config_file

        with open(config_path, "r") as stream:
            opt_dict = yaml.load(stream)

        # convert dictionary to attribute-like object
        opt = Munch(opt_dict)

        if self.running_mode.lower() != "train":
            # validate the existence of the model
            if not os.path.exists(opt.load_trained_model["path"]):
                model_path = Path(local_dir) / Path(self.config_file) / "latest.pth"
                zoo_client.download_model(opt.load_trained_model["path"], model_path)
                opt.load_trained_model["path"] = str(model_path)

        # load and validation training config
        if self.running_mode.lower() == "train":
            opt.isTrain = True

            # create a unique job name for this run
            job_name = self.get_job_name(opt.name)

            # create the job directory under the result folder
            opt.resultroot = Path(opt.save["results_folder"]) / Path(job_name)
            opt.checkpoints_dir = opt.resultroot / "checkpoints"
            os.mkdir(opt.resultroot)
            os.mkdir(opt.checkpoints_dir)
            if opt.save["save_training_inspections"]:
                opt.sample_dir = opt.resultroot / "samples"
                os.mkdir(opt.sample_dir)

            # copy the config file into the job directory
            shutil.copy(config_path, opt.resultroot)

            # default vaue for train_num
            opt.training_setting["train_num"] = -1

            # path to load offset from stage 1 of Auto-Alignment module
            opt.offsetlogpath = opt.resultroot / Path("offsets.log")

            # determine how to resize source image
            opt.resizeA = "toB"

        elif self.running_mode.lower() == "validation":
            opt.isTrain = False

            # check output folder exists
            opt.output_path = Path(opt.datapath["prediction"])
            if not opt.output_path.exists():
                opt.output_path.mkdir(parent=True)

            # copy the config file into the prediction directory
            shutil.copy(config_path, opt.output_path)

            # determine how to resize source image
            opt.resizeA = "toB"

        elif self.running_mode.lower() == "inference":
            opt.isTrain = False

            # check output folder exists
            opt.output_path = Path(opt.datapath["prediction"])
            if not opt.output_path.exists():
                opt.output_path.mkdir(parent=True)

            # copy the config file into the prediction directory
            shutil.copy(config_path, opt.output_path)

            # determine how to resize source image
            opt.resizeA = "ratio"

        else:
            raise NotImplementedError("mode name errir")

        # check validity of parameters and filing default values
        # TODO: add all checks on required parameters
        assert len(opt.network["input_patch_size"]) == 3

        opt.mode = self.running_mode.lower()
        self.print_options(opt)
        return opt
