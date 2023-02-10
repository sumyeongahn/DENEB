from tqdm import tqdm
import os
import copy
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import seaborn as sns

from ray import tune

from utils.avgmeter import *

from module.data.data import *
from module.net.net import *
from module.scores.margin import *
from module.etc.ema  import *

from alg.vanilla import Learner as BaseLearner
from utils.plot_util  import scatter_hist as scatter
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM
from sklearn.manifold import TSNE
from module.loss.gce import GeneralizedCELoss

from module.extract.feature import get_features


class Learner(BaseLearner):
    def __init__(self,args, second_stage=False, use_log_dir=False):
        super(Learner, self).__init__(args)
        self.train_loader, self.valid_loader, self.test_loader = get_loader(args, drop_last=False)
        if second_stage:
            
            if use_log_dir:
                indices = t.load(f'{self.args.log_dir}{self.args.uuid}_{self.args.seed}.pt')
            else:
                indices = t.load(f'{self.args.preset_dir}{self.args.uuid}_{self.args.seed}.pt')
                
            self.train_loader.dataset.cleansing(indices)
        

    

    # * ---- Ours ---- * #
    def train(self, tune_save= False):
        self.set_meter()

        model_b, opt_b, sch_b = get_model(self.args)
        model_d, opt_d, sch_d = get_model(self.args)
        
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        gce_fn = GeneralizedCELoss(reduction='none', q=float(self.args.q))
        
        logs_d = {}
        logs_b={}
        
        from sklearn.mixture import GaussianMixture as GMM
        self.set_meter()
        

        if not self.args.pretrained_bmodel:
            with tqdm(range(self.args.epochs)) as progress:
                for _, self.curr_epoch in enumerate(progress):
                    progress.set_description_str(f'Train epoch')
                    logs_b[self.curr_epoch] = self.log_init()
                    if self.curr_epoch < int(self.args.warmup):
                        model_b.train()
                        for batch_idx, data_tuple in enumerate(self.train_loader):
                            data = data_tuple[0].to(self.device)
                            label = data_tuple[1].to(self.device)
                            idx = data_tuple[4]

                            logit_b = model_b(data)
                            loss_b = loss_fn(logit_b, label) 
                            # loss_b = gce_fn(logit_b, label) 
                            loss_b_mean = loss_b.mean()
                            opt_b.zero_grad()
                            loss_b_mean.backward()
                            opt_b.step()

                            acc_b_mean = t.mean((logit_b.max(1)[1]==label).float().detach().cpu())
                            logs_b[self.curr_epoch]['train_acc'].update(acc_b_mean.detach().cpu(),len(label))
                            logs_b[self.curr_epoch]['train_loss'].update(loss_b_mean.detach().cpu(), len(label))

                            
                    else:
                        model_b.eval()
                        all_loss_b = t.zeros(len(self.train_loader.dataset))
                        
                        for batch_idx, data_tuple in enumerate(self.train_loader):
                            
                            data = data_tuple[0].to(self.device)
                            label = data_tuple[1].to(self.device)
                            idx = data_tuple[4]

                            logit_b = model_b(data)
                            loss_b = loss_fn(logit_b, label) 
                            # loss_b = gce_fn(logit_b, label) 
                            all_loss_b[idx] = loss_b.detach().cpu()
                        
                        all_loss_b = (all_loss_b - all_loss_b.min())/(all_loss_b.max() - all_loss_b.min())
                        all_loss_b = all_loss_b.reshape(-1,1)
                        gmm_b = GMM(n_components=2, max_iter = 10, reg_covar = 5e-4, tol=1e-2)
                        gmm_b.fit(all_loss_b)
                        prob_b = gmm_b.predict_proba(all_loss_b)
                        prob_b = t.tensor(prob_b[:, gmm_b.means_.argmin()])
                        label_pred_b = t.where(prob_b > float(self.args.p_threshold))[0]

                        if len(label_pred_b) == 0:
                            label_pred_b = t.arange(len(prob_b))
                            
                        labeled_loader_b=get_split_loader(label_pred_b,
                                                        prob_b[label_pred_b],
                                                        self.args,
                                                        unlabel=False)
                        
                        # Model 1 train
                        model_b.train()
                        for batch_idx, data_tuple in enumerate(labeled_loader_b):
                            idx = data_tuple[5]
                            data = data_tuple[0].to(self.device)
                            label = data_tuple[4].to(self.device)

                            logit_b = model_b(data)
                            loss = loss_fn(logit_b, label)
                            # loss = gce_fn(logit_b, label)
                            loss_b_mean = loss.mean()

                            opt_b.zero_grad()
                            loss_b_mean.backward()
                            opt_b.step()

            
                            acc_b_mean = t.mean((logit_b.max(1)[1]==label).float().detach().cpu())
                            logs_b[self.curr_epoch]['train_acc'].update(acc_b_mean.detach().cpu(),len(label))
                            logs_b[self.curr_epoch]['train_loss'].update(loss_b_mean.detach().cpu(), len(label))

                    sch_b.step()
                    logs_b = self.validation(model_b, loss_fn, sch_b, logs_b, model_name = 'Bias training')
            # model save
            t.save(model_b.state_dict(), f'./backup/model_{self.args.seed}.pt')
        model_b.load_state_dict(t.load(f'./backup/model_{self.args.seed}.pt'))

        
        entropy = t.zeros(len(self.train_loader.dataset))
        model_b.eval()
        for _, data_tuple in enumerate(self.train_loader):
            data = data_tuple[0].to(self.device)
            label = data_tuple[1].to(self.device)
            gtlabel = data_tuple[2]
            idx = data_tuple[4]
            logit_b = model_b(data) / float(self.args.tau)
            entropy[idx] = -t.sum((F.softmax(logit_b,dim=1)*F.log_softmax(logit_b,dim=1)).detach().cpu(), dim=1)
            
        
        

        entropy /= t.sum(entropy)
        self.train_loader.dataset.update_prob(entropy, log_print=True)
        self.train_loader.dataset.prob_sample_on()
        t.save(entropy, f'./backup/{self.args.seed}.pt')
            
        self.set_meter()

        with tqdm(range(self.args.epochs)) as progress:
            for _, self.curr_epoch in enumerate(progress):
                progress.set_description_str(f'Train epoch')
                logs_d[self.curr_epoch] = self.log_init()
                
                model_d.train()
                
                # Train confident samples
                for _, data_tuple in enumerate(self.train_loader):
                    
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    gtlabel = data_tuple[2]
                    idx = data_tuple[4]

                    logit_d = model_d(data)

                    # loss_d = loss_fn(logit_d, label)
                    loss_d = gce_fn(logit_d, label)
                    loss_d_mean = loss_d.mean()

                    opt_d.zero_grad()
                    loss_d_mean.backward()
                    opt_d.step()

                    acc_d_mean = t.mean((logit_d.max(1)[1]==label).float().detach().cpu())
                    logs_d[self.curr_epoch]['train_acc'].update(acc_d_mean.detach().cpu(),len(label))
                    logs_d[self.curr_epoch]['train_loss'].update(loss_d_mean.detach().cpu(), len(label))

                sch_d.step()
                logs_d = self.validation(model_d, loss_fn, sch_d, logs_d, model_name = 'debias training')
                if tune_save:
                    tune.report(loss= float(logs_d[self.curr_epoch]['valid_loss'].avg), accuracy = float(logs_d[self.curr_epoch]['valid_acc'].avg), test_accuracy = float(logs_d[self.curr_epoch]['test_acc'].avg), uuid = self.args.uuid)
