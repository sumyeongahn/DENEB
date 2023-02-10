import torch as t
import numpy as np
import random

def reproduce(args):

    # Random seed
    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    t.cuda.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)

    return args