import os
import shutil
import json
import argparse
from util.utils import ensure_dirs


class LatentGANConfig(object):
    """Base class of Config, provide necessary hyperparameters.
    """

    def __init__(self):
        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        phase = "train" if args.train else "test"
        self.is_train = phase == "train"

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        if self.exp_name is None:
            self.exp_name = "pqnet-PartNet-{}".format(args.category)
        self.exp_dir = os.path.join(args.proj_dir, self.exp_name)
        self.log_dir = os.path.join(self.exp_dir, 'log_{}'.format('lgan'))
        self.model_dir = os.path.join(self.exp_dir, 'model_{}'.format('lgan'))

        if phase == "train" and args.cont is not True and os.path.exists(self.log_dir):
            response = input('Experiment log/model already exists, overwrite to retrain? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.log_dir)
            shutil.rmtree(self.model_dir)

        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        self.parallel = False
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
            if len(str(args.gpu_ids).split(',')) > 1:
                self.parallel = True

        # save this configuration
        if self.is_train:
            with open(os.path.join(self.log_dir, 'config_{}.txt'.format('lgan')), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()

        # basic configuration
        self._add_basic_config_(parser)

        # dataset configuration
        self._add_dataset_config_(parser)

        # model configuration
        self._add_network_config_(parser)

        # training configuration
        self._add_training_config_(parser)

        # if not self.is_train:
        # testing configuration
        self._add_testing_config_(parser)

        # additional parameters if needed
        pass

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="proj_log",
                           help="path to project folder where experiment logs/models will be saved")
        group.add_argument('--exp_name', type=str, default=None, help="name of this experiment. "
                           "Automatically generated based on data category if not provided.")
        group.add_argument('-g', '--gpu_ids', type=str, default="0",
                           help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

        # phase
        mgroup  = parser.add_mutually_exclusive_group(required=True)
        mgroup.add_argument('--train', action='store_true', help="enter training phase for model training")
        mgroup.add_argument('--test', action='store_true', help="enter testing phase for generating fakes")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""
        group = parser.add_argument_group('dataset')
        group.add_argument('--data_root', type=str, help="file path to data", required=True)
        group.add_argument('--category', type=str, default="Chair", choices=['Chair', 'Table', 'Lamp'],
                           help="shape category name")
        group.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('lgan')
        group.add_argument('--n_dim', type=int, default=128, help='dimension of noise vector')
        group.add_argument('--h_dim', type=int, default=2048, help='dimension of MLP hidden layer ')
        group.add_argument('--z_dim', type=int, default=1024, help='dimension of shape code')

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--n_iters', type=int, default=100000, help="total number of training iterations")
        group.add_argument('--batch_size', type=int, default=64, help="batch size")
        group.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
        group.add_argument('--beta1', type=float, default=0.5, help="beta1 for Adam optimizer")
        group.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
        group.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        group.add_argument('--save_frequency', type=int, default=10000, help="save models every x iteration")

        # wgan-gp training
        group.add_argument('--critic_iters', type=int, default=1, help="critic iterations per generator iteration")
        group.add_argument('--gp_lambda', type=float, default=10, help="weight factor gradient penalty")

    def _add_testing_config_(self, parser):
        group = parser.add_argument_group('testing')
        group.add_argument('--n_samples', type=int, default=100, help="number of samples to generate when testing")
