__author__ = 'sumyeong ahn'
__email__ = 'sumyeongahn@kaist.ac.kr'

from datetime import datetime
import logging
from importlib import reload


def level_info(log_level):
    if log_level == 'debug':
        return logging.DEBUG
    elif log_level == 'info':
        return logging.INFO
    elif log_level == 'warning':
        return logging.WARNING
    elif log_level == 'error':
        return logging.ERROR
    elif log_level == 'critical':
        return logging.CRITICAL

def logger(args, file = True):
    logging.shutdown()
    reload(logging)

    __logger = logging.getLogger('Debiasing feature bias with noisy labels')
    __logger.setLevel(level_info('debug'))
    formatter = logging.Formatter('(%(asctime)s): %(message)s')

    # Stream handler
    sthandler = logging.StreamHandler()
    sthandler.setFormatter(formatter)
    sthandler.setLevel(level_info('info'))
    __logger.addHandler(sthandler)
    
    if file:
        # File hanlder
        mode = 'w+'
        fname = f'{args.log_dir}/{args.seed}_clean.log' if args.clean_valid else f'{args.log_dir}/{args.seed}.log'
        fhandler = logging.FileHandler(fname, mode=mode)    
        fhandler.setFormatter(formatter)
        fhandler.setLevel(level_info('debug'))
        __logger.addHandler(fhandler)

    return __logger
        
