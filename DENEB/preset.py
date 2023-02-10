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




def denoise_run(args, tune=True):
    if args.alg == 'vanilla':
        from alg.vanilla import Learner
        learner = Learner(args)
    
    elif args.alg == 'gce':
        from alg.gce import Learner
        learner = Learner(args)
    
    elif args.alg == 'sce':
        from alg.sce import Learner
        learner = Learner(args)
    
    elif args.alg == 'elr':
        from alg.elr import Learner
        learner = Learner(args)
    
    elif args.alg == 'aum':
        from alg.aum import Learner
        learner = Learner(args)

    elif args.alg == 'coteaching':
        from alg.coteaching import Learner
        learner = Learner(args)
    
    elif args.alg == 'coteaching+':
        from alg.coteaching_plus import Learner
        learner = Learner(args)

    elif args.alg == 'dividemix':
        from alg.dividemix import Learner
        learner = Learner(args)

    elif args.alg == 'fdividemix':
        from alg.fdividemix import Learner
        learner = Learner(args)
    
    learner.train(tune_save = tune)

def debias_run(args, tune=True):
    if args.alg == 'lff':
        from alg.lff import Learner
        learner = Learner(args)
    
    elif args.alg == 'jtt':
        from alg.jtt import Learner
        learner = Learner(args)
    
    elif args.alg == 'disen':
        from alg.disen import Learner
        learner = Learner(args)
    
    elif args.alg == 'deneb':
        from alg.deneb import Learner
        learner = Learner(args)
    
    learner.train(tune_save =tune)
    


def run(args, config):
    tune = True
    args.logger = print
    args.uuid = uuid.uuid1()

    # Reproducibility (Seed fix)
    args = reproduce(args)

    # Set preset index
    args = get_search(args,config)
    if args.alg in ['lff', 'jtt', 'disen', 'deneb']:
        debias_run(args, tune = tune)
    else:
        denoise_run(args, tune = tune)
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Dataset Bias wit Noisy Lables (DBwNL)')

    parser.add_argument('--num_workers',help='workers number',default=8,type=int)

    parser.add_argument('--model',help='Model architecture',default='CONV',type=str)
    parser.add_argument('--gpu',help='GPU',default='0',type=str)
    parser.add_argument('--dataset',help='Datasets',default='colored_mnist',type=str)
    parser.add_argument('--num_classes',help='Number of clsses', default=10,type=int)
    parser.add_argument('--epochs',help='Running epochs', default=100,type=int)
    parser.add_argument('--search_trial',help='Search trials', default=10,type=int)
    parser.add_argument('--seed',help='Seed', default=0,type=int)
    
    parser.add_argument('--lr_decay',help='LR decay method', default='StepLR',type=str)
    parser.add_argument('--lr',help='LR decay method', default=0.1,type=float)
    parser.add_argument('--batch_size',help='LR decay method', default=256,type=int)
    parser.add_argument('--lr_decay_rate',help='LR decay method', default=0.1,type=float)
    parser.add_argument('--momentum',help='LR decay method', default=0.9,type=float)
    parser.add_argument('--weight_decay',help='LR decay method', default=0.001,type=float)
    parser.add_argument('--lr_decay_opt',help='LR decay method', default=30,type=int)
    parser.add_argument('--opt',help='Optimizer', default='SGD',type=str)

    parser.add_argument('--nratio',help='Noise ratio', default=0.0,type=float)
    parser.add_argument('--noise_type',help='Noise type (Sym, Asym)',default='sym',type=str)
    
    parser.add_argument('--alg',help='Algorithm',default='none',type=str)
    parser.add_argument('--bratio',help='Bias ratio', default=0.01,type=float)
    parser.add_argument('--option',help='Option', default='null', type=str)

    parser.add_argument('--clean_valid', help='Clean balanced validationc option', action='store_true', default=False)
    parser.add_argument('--hyperopt', help='Use HyperOpt', action='store_true', default=False)

    parser.add_argument('--gpu_usage',help='Search gpu usage', default=0.5,type=float)
    parser.add_argument('--log_dir', help='log directory', default='./log/', type=str)
    parser.add_argument('--result_dir', help='log directory', default='none', type=str)
    parser.add_argument('--data_dir', help='path for loading data', default='../dataset/', type=str)

    args, unknown = parser.parse_known_args()
    
    #--- Setting ---#
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    curr_dir = os.getcwd() + '/'
    args.data_dir = curr_dir + args.data_dir
    args.dataset_logdir = f'{args.dataset}_clean' if args.clean_valid else f'{args.dataset}'
    args.log_dir = f'{curr_dir}/{args.log_dir}/{args.dataset_logdir}/{args.alg}/{args.nratio}_{args.bratio}/preset/'
    config = get_preset(args.alg, args.dataset)
    os.makedirs(args.log_dir, exist_ok=True)

    reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'test_accuracy', 'uuid', 'training_iteration'])
    

    if args.hyperopt:
        scheduler = ASHAScheduler(metric='test_accuracy', mode='max')
        search_alg = HyperOptSearch(config, metric='test_accuracy', mode='max')
        



    def trial_name_creator(trial):
        return f"{trial.trial_id}".split('_')[-1]

    if args.hyperopt:
        result = tune.run(partial(run, args), resources_per_trial={'cpu':args.num_workers, 'gpu':args.gpu_usage}, num_samples = args.search_trial, scheduler = scheduler, progress_reporter = reporter, trial_name_creator = trial_name_creator, local_dir =args.log_dir, search_alg=search_alg, max_failures=5)
        
        
    else:
        result = tune.run(partial(run, args), config=config, resources_per_trial={'cpu':args.num_workers, 'gpu':args.gpu_usage}, num_samples = args.search_trial, progress_reporter = reporter, trial_name_creator = trial_name_creator, local_dir =args.log_dir)
        
        
    df = result.dataframe()
    df.to_csv(args.log_dir+'conf_clean.log' if args.clean_valid else args.log_dir+'conf.log')
                
    