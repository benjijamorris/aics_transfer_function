import argparse
import os
from transfer_function.util import util
import shutil
import torch
from transfer_function.models.__init__ import get_option_setter
import yaml
import pdb


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self,isTrain):
        """Reset the class; indicates the class hasn't been initailized"""
        self.isTrain = isTrain
        self.opt = None

    def get_job_name(self,name,tag):
        import datetime
        now = datetime.datetime.now()
        time = now.strftime("%m%d_%H%M")
        try:
            import git             # pip3 install gitpython --user
            repo = git.Repo(search_parent_directories=True)
            gitsha = (repo.head.object.hexsha)[-4:]
        except:
            gitsha = 'xxxx'
        import uuid
        jobid = str(uuid.uuid4())
        wholename = "%s_%s_%s_%s_%s"%(time,gitsha,name,tag,jobid[-6:])
        return wholename

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--config', type=str, required=True, help='')
        parser.add_argument('--debug', action='store_true', help='')
        parser.add_argument('--gpu', default='', help='')
        opt, _ = parser.parse_known_args()
        with open(opt.config, 'r') as stream:
            argdict = yaml.load(stream)
        if not 'tag' in argdict:
            argdict['tag'] = ''
        if not 'seed' in argdict:
            argdict['seed'] = 0
        if not 'stn_adjust_fixed_z' in argdict:
            argdict['stn_adjust_fixed_z'] = False
        if not 'readoffsetfrom' in argdict:
            argdict['readoffsetfrom'] = ''
        if not 'align_all_axis' in argdict and argdict['stn_adjust_fixed_z']:
            print(f'WARNING: key align_all_axis is not in the yaml file. Set False (i.e. only align z axis)')
        for key in argdict.keys():
            vars(opt)[key]=argdict[key]        
        if opt.gpu != '':
            opt.gpu_ids = opt.gpu
        self.opt = opt
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        if opt.continue_from != '' and opt.isTrain:
            opt.continue_train = True
        else:
            opt.continue_train = False
        opt.phase = 'train' if self.isTrain else 'test'

        if self.isTrain:
            opt.job_name = self.get_job_name(opt.name,opt.tag)
            opt.resultroot = opt.results_folder + '/' + opt.job_name + '/'
            util.mkdir(opt.resultroot)
            util.mkdir(opt.resultroot + 'checkpoints/')
            opt.checkpoints_dir = opt.resultroot + 'checkpoints/'
            opt.sample_dir = opt.resultroot + 'samples/'
            opt.offsetlogpath = opt.resultroot + 'offsets.log'
            if opt.save_training_process:
                util.mkdir(opt.sample_dir)
            shutil.copy(opt.config,opt.resultroot)
        else:
            opt.checkpoints_dir = opt.continue_from + 'checkpoints/' 
            if opt.save_training_process:
                opt.sample_dir = opt.continue_from + 'samples/' 

        self.print_options(opt)

        # set gpu ids
        str_ids = str(opt.gpu_ids).split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        size_in = str(opt.size_in).split(',')
        assert len(size_in) == 3
        opt.size_in = (int(size_in[0]),int(size_in[1]),int(size_in[2]))
        self.opt = opt
        return self.opt