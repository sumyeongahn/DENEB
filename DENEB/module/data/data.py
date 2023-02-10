import copy
from torch.utils.data import DataLoader
import torch as t
import numpy as np
import random
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import DataLoader
transforms = {
    
    "cifar10": {
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },

    "colored_mnist": {
        "train": T.Compose([]),
        "valid": T.Compose([]),
        "test": T.Compose([])
        },

    "biased_mnist": {
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(56, scale=(0.9,1.1)),
                T.ColorJitter(hue=0.05, saturation = 0.05),
                T.RandomRotation((-10,10)),
                T.ToTensor()
            ]
        ),
        "valid": T.Compose([]),
        "test": T.Compose([])
        },
    
    "corrupted_cifar": {
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    "bar": {
        "train": T.Compose(
        [
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    },
    
    "bffhq": {
        "train": T.Compose([
            T.ToPILImage(),
            T.RandomCrop(128, padding=4),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        },
}



def get_dataset(dataset, data_dir, split, bias, noise, noise_type, seed, log_print=True):
    if split in ['train','valid']:
        if dataset in ['colored_mnist','biased_mnist', 'corrupted_cifar', 'cifar10']:
            tmp= "%s/%s/%s_bias_%s.pt" %(data_dir,dataset,dataset,str(bias))
            if log_print:
                print('%s data: '%(split),tmp)
            data = t.load(tmp)[split]
        elif dataset in ['bar','bffhq']:
            tmp= "%s/%s/%s.pt" %(data_dir,dataset,dataset)
            if log_print:
                print('%s data: '%(split),tmp)
            data = t.load(tmp)[split]
        else:
            if log_print:
                print('Wrong dataset')
            import sys
            sys.exit(0)
    elif split == 'test':
        tmp = "%s/%s/%s_test.pt" %(data_dir,dataset,dataset)
        if log_print:
                print('%s data: '%(split),tmp)
        data = t.load(tmp)

    transform = transforms[dataset][split]    
    return loader(data,noise,split,transform, noise_type, seed,  log_print=log_print)

class loader(DataLoader):
    def __init__(self,data,noise,split,transform,noise_type,seed, log_print = True):
        self.fix_seed(seed)
        self.transform = transform
        self.data = data['data'].float().clone()
        self.label = data['label'].long().clone()
        self.gt_label = data['label'].clone()
        self.b_label = data['b_label'].clone()

        self.prob_on = False
        if split !='test':
            if noise_type == 'sym':
                self.noise_sym(noise)
            elif noise_type == 'asym':
                self.noise_asym(noise)
            else:
                raise NotImplementedError

        self.data_backup = self.data.clone()
        self.label_backup = self.label.clone()
        self.gt_label_backup = self.gt_label.clone()
        self.b_label_backup = self.b_label.clone()

        self.prob = t.zeros(len(self.label))
        self.prob_backup = self.prob.clone()

        major_pos = t.where((self.gt_label == self.label) & (self.gt_label == self.b_label))[0]
        minor_pos = t.where((self.gt_label == self.label) & (self.gt_label != self.b_label))[0]
        major_neg = t.where((self.gt_label != self.label) & (self.gt_label == self.b_label))[0]
        minor_neg = t.where((self.gt_label != self.label) & (self.gt_label != self.b_label))[0]
        if log_print:
            print(len(major_pos), len(minor_pos), len(major_neg), len(minor_neg))
        
    def __getitem__(self,idx):
        idx = self.idx_sample() if self.prob_on else idx
        if type(self.b_label) == dict:
            return self.transform(self.data[idx]), self.label[idx], self.gt_label[idx], -1 , idx
        
        return self.transform(self.data[idx]), self.label[idx], self.gt_label[idx], self.b_label[idx], idx
    
    def __len__(self):
        return len(self.label)

    def idx_sample(self):
        return t.clamp(t.sum(t.rand(1)>self.prob), 0, len(self.label)-1 )

    def fix_seed(self,seed=888):
        np.random.seed(seed)
        random.seed(seed)
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        np.random.seed(seed)

    def noise_sym(self,noise):
        for c in range(int(t.max(self.label))+1):
            pos = t.where(self.label == c)[0]
            randperm = t.randperm(len(pos))
            pos = pos[randperm]
            pos = pos[:int(len(pos) * noise)]
            new_label = t.randint(int(t.max(self.label)+1), (len(pos),))
            self.label[pos] = new_label


    def noise_asym(self, noise):
        self.fix_seed()

        for i in range(int(t.max(self.label)+1)):
            indices = np.where(self.label == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < noise * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.label[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.label[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.label[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.label[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.label[idx] = 7

    def update_prob(self,prob, log_print=False):
        if log_print:
            major_pos = t.where((self.gt_label == self.label) & (self.gt_label == self.b_label))[0]
            minor_pos = t.where((self.gt_label == self.label) & (self.gt_label != self.b_label))[0]
            major_neg = t.where((self.gt_label != self.label) & (self.gt_label == self.b_label))[0]
            minor_neg = t.where((self.gt_label != self.label) & (self.gt_label != self.b_label))[0]
            print('-'*30)
            print('Major Positive: %f' %(t.sum(prob[major_pos])))
            print('Minor Positive: %f' %(t.sum(prob[minor_pos])))
            print('Major Negative: %f' %(t.sum(prob[major_neg])))
            print('Minor Negative: %f' %(t.sum(prob[minor_neg])))
            print('-'*30)
        self.prob = t.cumsum(prob,dim=0)
        
    def prob_sample_on(self):
        self.prob_on = True
    
    def prob_sample_off(self):
        self.prob_on = False

    def statistics_print(self,pos=None):
        if pos == None:
            major_pos = t.where((self.gt_label == self.label) & (self.gt_label == self.b_label))[0]
            minor_pos = t.where((self.gt_label == self.label) & (self.gt_label != self.b_label))[0]
            major_neg = t.where((self.gt_label != self.label) & (self.gt_label == self.b_label))[0]
            minor_neg = t.where((self.gt_label != self.label) & (self.gt_label != self.b_label))[0]
            print('-'*30)
            print('Major Positive: %d' %(len(major_pos)))
            print('Minor Positive: %d' %(len(minor_pos)))
            print('Major Negative: %d' %(len(major_neg)))
            print('Minor Negative: %d' %(len(minor_neg)))
            print('-'*30)
        else:
            label = self.label[pos]
            gt_label = self.gt_label[pos]
            b_label = self.b_label[pos]
            prob = self.prob[pos]

            major_pos = t.where((gt_label == label) & (gt_label == b_label))[0]
            minor_pos = t.where((gt_label == label) & (gt_label != b_label))[0]
            major_neg = t.where((gt_label != label) & (gt_label == b_label))[0]
            minor_neg = t.where((gt_label != label) & (gt_label != b_label))[0]
            print('-'*30)
            print('Major Positive: %d' %(len(major_pos)))
            print('Minor Positive: %d' %(len(minor_pos)))
            print('Major Negative: %d' %(len(major_neg)))
            print('Minor Negative: %d' %(len(minor_neg)))
            print('-'*30)

    def cleansing(self,pos):
        self.data = self.data[pos]
        self.label = self.label[pos]
        self.gt_label = self.gt_label[pos]
        self.b_label = self.b_label[pos]
        self.prob = self.prob[pos]
        
        self.statistics_print()
        
    def add_class(self):
        num_class = t.max(self.label_backup)+1
        num_threshold_samples = len(self.label) // (num_class+1)
        self.label[t.randperm(len(self.label))[:num_threshold_samples]] = num_class

    def data_renewal(self):
        self.data = self.data_backup.clone()
        self.label = self.label_backup.clone()
        self.gt_label = self.gt_label_backup.clone()
        self.b_label = self.b_label_backup.clone()
        self.prob = self.prob_backup.clone()

def get_loader(args, drop_last = True):
    train_dataset = get_dataset(dataset = args.dataset,
                                data_dir = args.data_dir,
                                split = 'train',
                                bias = args.bratio,
                                noise = args.nratio,
                                noise_type = args.noise_type,
                                seed=args.seed)
    valid_dataset = get_dataset(dataset = args.dataset,
                                data_dir = args.data_dir,
                                split = 'valid',
                                bias = args.bratio,
                                noise = args.nratio,
                                noise_type = args.noise_type,
                                seed=args.seed)
    test_dataset = get_dataset(dataset = args.dataset,
                                data_dir = args.data_dir,
                                split = 'test',
                                bias = args.bratio,
                                noise = args.nratio,
                                noise_type = args.noise_type,
                                seed=args.seed)
    
        
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory = True, drop_last = drop_last)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory = True, drop_last = False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory = True, drop_last = False)

    if args.clean_valid:
        valid_loader = copy.deepcopy(test_loader)
        valid_length = len(valid_loader.dataset)
        remain_size = int(valid_length * 0.5)
        remain_length = t.arange(remain_size)
        valid_loader.dataset.cleansing(remain_length)
        
        test_length = remain_length + remain_size
        test_loader.dataset.cleansing(test_length)

    args.logger('==> %s dataset...' %(args.dataset))
    args.logger('Train: %d, Valid: %d, Test: %d' %(len(train_dataset), len(valid_dataset), len(test_dataset)))
    args.logger('Bias: %.4f, Noise: %.4f'%(args.bratio, args.nratio))
    
    return train_loader, valid_loader, test_loader




class split_loader(DataLoader):
    def __init__(self,idx, prob, transform, args, unlabel):
        train_dataset = get_dataset(dataset = args.dataset,
                                data_dir = args.data_dir,
                                split = 'train',
                                bias = args.bratio,
                                noise = args.nratio,
                                noise_type = args.noise_type,
                                seed=args.seed,
                                log_print=False)

        train_dataset.statistics_print(idx)
        self.transform = transform
        self.prob = prob
        self.data = train_dataset.data[idx].clone()
        self.label = train_dataset.label[idx].clone()
        self.unlabel = unlabel
        self.idxs = idx

        
    def __getitem__(self,idx):
        if self.unlabel:
            return self.transform(self.data[idx]), self.transform(self.data[idx]), self.prob[idx], idx, self.idxs[idx]
        else:
            return self.transform(self.data[idx]), self.transform(self.data[idx]), self.prob[idx],  idx, self.label[idx] , self.idxs[idx]
            
    def __len__(self):
        return len(self.data)

def get_split_loader(idx, prob, args, unlabel):
    
    return DataLoader(split_loader(idx, prob, transforms[args.dataset]['train'], args, unlabel), 
                        batch_size = args.batch_size, 
                        shuffle=True, 
                        num_workers = args.num_workers, 
                        drop_last = True)



