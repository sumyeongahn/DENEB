import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.models as models
import torch.optim as optim


from module.net.models.MLP import *
from module.net.models.CONV import *
from module.net.models.ResNet import *
from module.net.models.CIFAR_ResNet import *
from module.net.models.PreActResNet import *
from module.net.models.WideResNet import *

def get_model(args):
    model_tag = args.model
    num_classes = args.num_classes
    device = args.device
    
    if model_tag == 'MLP':
        model = MLP(num_classes)
    elif model_tag == 'CONV':
        model = CONV(num_classes)
    elif model_tag == 'ResNet18':
        model = ResNet18(num_classes = num_classes)
    elif model_tag == 'pretrained_ResNet18':
        model = pretrained_ResNet18(num_classes = num_classes)
    elif model_tag == 'pretrained_ResNet34':
        model = pretrained_ResNet34(num_classes = num_classes)
    elif model_tag == 'pretrained_ResNet50':
        model = pretrained_ResNet50(num_classes = num_classes)
    elif model_tag == 'pretrained_ResNet101':
        model = pretrained_ResNet101(num_classes = num_classes)
    elif model_tag == 'CIFAR_ResNet18':
        model = CIFAR_ResNet18(num_classes = num_classes)
    elif model_tag == 'ResNet20':
        model = ResNet20(num_classes = num_classes)
    elif model_tag == 'ResNet34':
        model = ResNet34(num_classes = num_classes)
    elif model_tag == 'CIFAR_ResNet34':
        model = CIFAR_ResNet34(num_classes = num_classes)
    elif model_tag == 'ResNet50':
        model = ResNet50(num_classes = num_classes)
    elif model_tag == 'ResNet101':
        model = ResNet101(num_classes = num_classes)
    elif model_tag == 'ResNet152':
        model = ResNet152(num_classes = num_classes)
    elif model_tag == 'ResNet200':
        model = ResNet200(num_classes = num_classes)
    elif model_tag == 'PreActResNet18':
        model = PreActResNet18(num_classes = num_classes)
    else:
        raise NotImplementedError
    
    model = model.to(device)

    if args.dataset in ['bar', 'bffhq']:
        model = t.nn.DataParallel(model)


    args.logger("==> %s set..."%(model_tag))
    if args.opt == 'SGD':
        opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
        args.logger('SGD Optim: (lr = %.4f, weight decay = %.4f, momentum = %.4f)' %(args.lr, args.weight_decay, args.momentum))
    elif args.opt == 'Adam':
        opt = optim.Adam(model.parameters(), lr=args.lr)
        args.logger('Adam Optim: (lr = %.4f, weight decay = %.4f)' %(args.lr, args.weight_decay))
    else:
        raise NotImplementedError

    if args.lr_decay == 'StepLR':
        sch = optim.lr_scheduler.StepLR(opt, step_size = args.lr_decay_opt, gamma = args.lr_decay_rate)
        args.logger('StepLR sch: (Decay step = %d, gamma = %.4f)' %(args.lr_decay_opt, args.lr_decay_rate))
    elif args.lr_decay == 'MultiStepLR':
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones = args.lr_decay_opt, gamma = args.lr_decay_rate)
        args.logger('MultiStepLR sch: (Decay step = %s, gamma = %.4f)' %(args.lr_decay_opt, args.lr_decay_rate))
    elif args.lr_decay == 'CosineAnnealingLR':
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.epochs)
        args.logger('CosineAnnealingLR sch: (epochs: %d)' %( args.epochs))
    else:
        raise NotImplementedError


    

    return model, opt, sch
    
    