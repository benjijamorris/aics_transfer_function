import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """
    This class is an abstract base class (ABC) for models.
    """

    def __init__(self, opt):
        """
        Initialize the BaseModel class.

        Parameters
        -----------------
            opt: a subclass of BaseOptions
                stores all the experiment flags;
                needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):
            -- self.model_names (str list):
            -- self.visual_names (str list):
            -- self.optimizers (optimizer list):
        """
        self.opt = opt
        self.gpu_ids = [0]
        self.isTrain = opt.isTrain
        self.device = torch.device("cuda:0")
        if opt.mode == "train":
            self.save_dir = opt.checkpoints_dir
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward pass;
        called by both functions <optimize_parameters> and <test>.
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights;
        called in every training iteration
        """
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be
            a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]
        if not self.isTrain or opt.load_trained_model["path"] is not None:
            if opt.load_trained_model["load_iter"] > 0:
                load_suffix = "iter_%s" % str(opt.load_trained_model["load_iter"])
            else:
                load_suffix = str(opt.load_trained_model["epoch"])
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save
        intermediate steps for backprop. It also calls <compute_visuals> to
        produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """
        Update learning rates for all the networks;
        called at the end of every epoch
        """
        for scheduler in self.schedulers:
            if self.opt.training_setting["lr_policy"] == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def get_current_visuals(self):
        """
        Return visualization images. train.py will display these images
        with visdom, and save the images to a HTML
        """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """
        Return traning losses / errors. train.py will print out these errors
        on console, and save them to a file
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                try:
                    errors_ret[name] = float(getattr(self, "loss_" + name))
                except Exception as e:
                    print(e)
                    errors_ret[name] = 0
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name
            '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name
            '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            try:
                if isinstance(name, str):
                    mp = self.opt.load_trained_model["path"]
                    load_filename = f"{mp}/{epoch}_net_{name}.pth"
                    load_path = load_filename
                    net = getattr(self, "net" + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print("loading the model from %s" % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, "_metadata"):
                        del state_dict._metadata

                    # patch InstanceNorm checkpoints prior to 0.4
                    # need to copy keys here because we mutate in loop
                    for key in list(state_dict.keys()):
                        self.__patch_instance_norm_state_dict(
                            state_dict, net, key.split(".")
                        )
                    net.load_state_dict(state_dict)
            except Exception as e:
                print(f"error {e}, when loading models")
                print(f"model not found from {name}")
                if not self.isTrain:
                    raise ValueError("model not found\n")

    def print_networks(self, verbose):
        """
        Print the total number of parameters in the network and (if verbose)
        network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(f"Network {name}: ")
                print(f"Total number of parameters: {num_params}")
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
