import torch as t
import os, argparse
from functools import partial
import pandas as pd
import uuid

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

# Custom
from utils.conf import *
from utils.logger import *
from utils.reproduce import *
from utils.search_preset import *


t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False



def denoise_run(args, alg, second_stage, tune=True):
    if alg == 'vanilla':
        from alg.vanilla import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    elif alg == 'gce':
        from alg.gce import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    elif alg == 'sce':
        from alg.sce import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    elif alg == 'elr':
        from alg.elr import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    elif alg == 'aum':
        from alg.aum import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)

    elif alg == 'coteaching':
        from alg.coteaching import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    elif alg == 'coteaching+':
        from alg.coteaching_plus import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)

    elif alg == 'dividemix':
        from alg.dividemix import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)

    elif alg == 'fdividemix':
        from alg.fdividemix import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    learner.train(tune_save = tune)

def debias_run(args, alg, second_stage, tune=True):
    if alg == 'lff':
        from alg.lff import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    elif alg == 'jtt':
        from alg.jtt import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    elif alg == 'disen':
        from alg.disen import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    elif alg == 'deneb':
        from alg.deneb import Learner
        learner = Learner(args, second_stage=second_stage, use_log_dir = True)
    
    learner.train(tune_save =tune)
    
def run(args, alg, stage):
    args.log_dir = f"{args._log_dir}/{args.seed}/"
    os.makedirs(args.log_dir, exist_ok = True)
    tune=False
    # Logger
    _logger = logger(args, file=True)
    args.logger = _logger.info
    args.logger(args)
    
    # Reproducibility (Seed fix)
    args = reproduce(args)
    print(alg)
    second_stage = True if stage == 'second' else False
    if alg in ['lff', 'jtt', 'disen', 'deneb']:
        debias_run(args, alg, second_stage = second_stage, tune = tune)
    else:
        denoise_run(args, alg, second_stage = second_stage, tune = tune)
    






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Dataset Bias wit Noisy Lables (DBwNL)')

    parser.add_argument('--num_workers',help='workers number',default=8,type=int)

    parser.add_argument('--model',help='Model architecture',default='CONV',type=str)
    parser.add_argument('--gpu',help='GPU',default='0',type=str)
    parser.add_argument('--dataset',help='Datasets',default='colored_mnist',type=str)
    parser.add_argument('--num_classes',help='Number of clsses', default=10,type=int)
    parser.add_argument('--epochs',help='Running epochs', default=100,type=int)
    parser.add_argument('--seed',help='Seed', default=0,type=int)
    parser.add_argument('--indep_run',help='independent run', default=3,type=int)
    
    parser.add_argument('--lr_decay',help='LR decay method', default='StepLR',type=str)
    parser.add_argument('--lr',help='LR decay method', default=0.1,type=float)
    parser.add_argument('--batch_size',help='LR decay method', default=256,type=int)
    parser.add_argument('--lr_decay_rate',help='LR decay method', default=0.1,type=float)
    parser.add_argument('--momentum',help='LR decay method', default=0.9,type=float)
    parser.add_argument('--weight_decay',help='LR decay method', default=0.001,type=float)
    parser.add_argument('--lr_decay_opt',help='LR decay method', default=30,type=int)
    parser.add_argument('--opt',help='Optimizer', default='SGD',type=str)

    parser.add_argument('--alg',help='Denoise algorithm',default='none',type=str)
    parser.add_argument('--nratio',help='Noise ratio', default=0.0,type=float)
    parser.add_argument('--noise_type',help='Noise type (Sym, Asym)',default='sym',type=str)
    
    parser.add_argument('--bratio',help='Bias ratio', default=0.01,type=float)
    parser.add_argument('--option',help='Option', default='null', type=str)

    parser.add_argument('--clean_valid', help='Clean balanced validationc option', action='store_true', default=False)
    parser.add_argument('--resampling', help='Use resampling', action='store_true', default=False)

    parser.add_argument('--gpu_usage',help='Search gpu usage', default=0.5,type=float)
    parser.add_argument('--log_dir', help='log directory', default='./log/', type=str)
    parser.add_argument('--data_dir', help='path for loading data', default='../dataset/', type=str)
    parser.add_argument('--result_dir', help='Preset directory', default='none')
    
    args, unknown = parser.parse_known_args()
    
    first_alg = args.alg.split('_')[0]
    second_alg = args.alg.split('_')[1]
    #--- Setting ---#
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    curr_dir = os.getcwd() + '/'
    args.data_dir = curr_dir + args.data_dir
    args.dataset_logdir = f'{args.dataset}_clean' if args.clean_valid else f'{args.dataset}'
    
    
    args.first_dir = f'{curr_dir}/{args.log_dir}/{args.dataset_logdir}/{first_alg}/{args.nratio}_{args.bratio}/'
    args.second_dir = f'{curr_dir}/{args.log_dir}/{args.dataset_logdir}/{second_alg}/{args.nratio}_{args.bratio}/'
    
    args._log_dir = f'{curr_dir}/{args.log_dir}/{args.dataset_logdir}/{args.alg}/{args.nratio}_{args.bratio}/'

    if first_alg in ['lff', 'disen']:
        args.resampling = False
    else:
        args.resampling=True

    df = pd.read_csv(args.second_dir+'preset/'+'conf_clean.log' if args.clean_valid else args.second_dir+'preset/'+'conf.log', index_col=0)
    df = df.sort_values(by=['loss','accuracy'], ascending=[True,False]).iloc[0]
    args = inject(args, df.to_dict())
    
    run(args, alg = args.alg.split('_')[1], stage = 'second')

    