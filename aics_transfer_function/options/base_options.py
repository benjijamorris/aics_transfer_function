import os
from pathlib import Path
import shutil
import yaml
import datetime
import git  # pip3 install gitpython --user


class BaseOptions():
    """
    This class defines options used during both training and test time.
    """
    def __init__(self, config_file, isTrain):
        self.isTrain = isTrain
        self.config_file = config_file

    def get_job_name(self, name: str = "TF", tag: str = "default"):
        """
        Generate a unique name for this experiment

        Parameters
        -----------
        name: str
            the name of your experiment, e.g. 20xto100x, 100xtoSR, etc.

        tag: str
            a specific tag to be added to the experiment name, e.g., production

        Returns
        -----------
        fullname: str
            the full job name in the format of 
            jobname_jobtag_gitSHA_time
        """

        now = datetime.datetime.now()
        time = now.strftime("%m%d_%H%M")
        try:            
            repo = git.Repo(search_parent_directories=True)
            gitsha = (repo.head.object.hexsha)[-4:]
        except Exception:
            gitsha = 'NoGitSHA'

        fullname = f"{name}_{tag}_{gitsha}_{time}"
        return fullname

    def print_options(self, opt):
        """
        Print options
        """
        message = '\n----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            message += f"{str(k)}: {str(v)}\n"
        message += '\n----------------- End -------------------\n'
        print(message)

    def parse(self):
        """
        parse the option arguments and fill with default values when missing
        """

        with open(self.config_file, 'r') as stream:
            opt = yaml.load(stream)

        opt.isTrain = self.isTrain   # flag for train/test

        if opt.continue_from != '' and opt.isTrain:
            opt.continue_train = True
        else:
            opt.continue_train = False

        if self.isTrain:
            # create a unique job name for this run
            opt.job_name = self.get_job_name(opt.name, opt.tag)

            # create the job directory under the result folder
            opt.resultroot = Path(opt.results_folder) / Path(opt.job_name)
            opt.checkpoints_dir = opt.resultroot / "checkpoints"
            os.mkdir(opt.resultroot)
            os.mkdir(opt.checkpoints_dir)
            if opt.save_training_inspections:
                opt.sample_dir = opt.resultroot / "samples"
                os.mkdir(opt.sample_dir)

            # copy the config file into the job directory
            shutil.copy(opt.config, opt.resultroot)

            # TODO: check AA code
            opt.offsetlogpath = opt.resultroot + 'offsets.log' 

        # check validity of parameters and filing default values
        # TODO: add all checks
        opt.train_num = -1
        assert len(opt.size_in) == 3

        self.print_options(opt)
        return opt
