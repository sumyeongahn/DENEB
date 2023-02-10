from tqdm import tqdm
import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 

from ray import tune

from utils.avgmeter import *

from module.data.data import *
from module.net.net import *

class Learner(object):
    def __init__(self,args):
        self.device = t.device(args.device)
        self.args = args
        self.train_loader, self.valid_loader, self.test_loader = get_loader(args)


    def search(self):
        space = {
            "lr": hp.loguniform("lr", 1e-10, 0.1),
            "momentum": hp.uniform("momentum", 0.1, 0.9),
        }


    def set_meter(self):
        self.best_loss = np.inf
        self.curr_epoch = 0
        self.best_acc = 0
        self.best_epoch = 0

    def log_init(self):
        return {'train_acc': AvgMeter(),
                'train_loss': AvgMeter(),
                'valid_acc': AvgMeter(),
                'valid_loss': AvgMeter(),
                'test_acc': AvgMeter(),
                'test_major': AvgMeter(),
                'test_minor': AvgMeter(),
                'test_loss': AvgMeter(),}
    


    # * ---- Validation ---- * #
    def validation(self,model, loss_fn, sch, logs, model_name, do_update=True):
        model.eval()
        
        for batch_idx, data_tuple in enumerate(self.valid_loader):
            data = data_tuple[0].to(self.device)
            label = data_tuple[1].to(self.device)

            logit = model(data)
            loss = loss_fn(logit, label)
            
            loss_mean = loss.mean()
            acc_mean = t.mean((logit.max(1)[1]==label).float().detach().cpu())
            
            logs[self.curr_epoch]['valid_acc'].update(acc_mean.detach().cpu(),len(label))
            logs[self.curr_epoch]['valid_loss'].update(loss_mean.detach().cpu(), len(loss))

        for batch_idx, data_tuple in enumerate(self.test_loader):
            data = data_tuple[0].to(self.device)
            label = data_tuple[1].to(self.device)
            blabel = data_tuple[3].to(self.device)
            
            logit = model(data)
            loss = loss_fn(logit, label)

            loss_mean = loss.mean()
            acc = (logit.max(1)[1]==label).float().detach().cpu()
            major = t.where(label == blabel)[0]
            minor = t.where(label != blabel)[0]

            major_acc = acc[major]
            minor_acc = acc[minor]

            acc_mean = acc.mean()
            major_acc_mean = major_acc.mean()
            minor_acc_mean = minor_acc.mean()

            logs[self.curr_epoch]['test_acc'].update(acc_mean.detach().cpu(),len(label))
            logs[self.curr_epoch]['test_major'].update(major_acc_mean.detach().cpu(),len(major))
            logs[self.curr_epoch]['test_minor'].update(minor_acc_mean.detach().cpu(),len(minor))
            logs[self.curr_epoch]['test_loss'].update(loss_mean.detach().cpu(), len(loss))

        if logs[self.curr_epoch]['valid_acc'].avg >= self.best_acc and do_update:
            self.best_acc = logs[self.curr_epoch]['valid_acc'].avg
            self.best_epoch = self.curr_epoch
    
        self.args.logger(f'=> Option: {model_name}')
        self.args.logger(f'=> current stats (Curr epoch: {self.curr_epoch+1} / {self.args.epochs})')
        self.args.logger("Train Loss: %.4f \t Valid Loss: %.4f \t Test Loss: %.4f" 
                                            %(logs[self.curr_epoch]['train_loss'].avg,
                                                logs[self.curr_epoch]['valid_loss'].avg,
                                                logs[self.curr_epoch]['test_loss'].avg))
        self.args.logger("Train Acc: %.4f \t Valid Acc: %.4f \t Test Acc: %.4f" 
                                            %(logs[self.curr_epoch]['train_acc'].avg,
                                                logs[self.curr_epoch]['valid_acc'].avg,
                                                logs[self.curr_epoch]['test_acc'].avg))
        self.args.logger("Major Acc: %.4f \t Minor Acc: %.4f"  %(logs[self.curr_epoch]['test_major'].avg, logs[self.curr_epoch]['test_minor'].avg))
        self.args.logger(f'=> Best stats  (Best epoch: {self.best_epoch+1})')
        self.args.logger("Train Loss: %.4f \t Valid Loss: %.4f \t Test Loss: %.4f" 
                                            %(logs[self.best_epoch]['train_loss'].avg,
                                                logs[self.best_epoch]['valid_loss'].avg,
                                                logs[self.best_epoch]['test_loss'].avg))
        self.args.logger("Train Acc: %.4f \t Valid Acc: %.4f \t Test Acc: %.4f " 
                                            %(logs[self.best_epoch]['train_acc'].avg,
                                                logs[self.best_epoch]['valid_acc'].avg,
                                                logs[self.best_epoch]['test_acc'].avg))
        self.args.logger("Major Acc: %.4f\t Minor Acc: %.4f"%(logs[self.best_epoch]['test_major'].avg, logs[self.best_epoch]['test_minor'].avg) )

        if do_update:
            self.log_end(model_name, logs)
        self.log_statistics(sch)

        return logs

    def log_statistics(self, sch):
        value = sch.get_last_lr()[0]
        self.args.logger("Current learning rate: %.3f" %(value))

    def log_end(self,model_name, logs):
        if (self.curr_epoch+1) == self.args.epochs:
            import csv
            if self.args.result_dir == 'none':
                return
            
            csv_dir = f'{self.args.end_log_dir}'
            os.makedirs(csv_dir,exist_ok=True)
            if self.args.clean_valid:
                fname = f'{csv_dir}/result_clean.csv'    
            else:
                fname = f'{csv_dir}/result.csv'
            
            f = open(fname, 'a', newline ='')
            wr = csv.writer(f)
            wr.writerow([f'{self.args.seed}_End',100*float(logs[self.curr_epoch]['test_acc'].avg)])
            wr.writerow([f'{self.args.seed}_Best',100*float(logs[self.best_epoch]['test_acc'].avg)])
            wr.writerow([f'{self.args.seed}_End_major',100*float(logs[self.curr_epoch]['test_major'].avg)])
            wr.writerow([f'{self.args.seed}_Best_major',100*float(logs[self.best_epoch]['test_major'].avg)])
            wr.writerow([f'{self.args.seed}_End_minor',100*float(logs[self.curr_epoch]['test_minor'].avg)])
            wr.writerow([f'{self.args.seed}_Best_minor',100*float(logs[self.best_epoch]['test_minor'].avg)])
            f.close()

    # * ---- Vanilla ---- * #
    def train(self, tune_save= False):
        
        model, opt, sch = get_model(self.args)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        logs = {}

        self.set_meter()

        with tqdm(range(self.args.epochs)) as progress:
            for _, self.curr_epoch in enumerate(progress):
                progress.set_description_str(f'Train epoch')
                model.train()
                logs[self.curr_epoch] = self.log_init()
                    
                for _, data_tuple in enumerate(self.train_loader):
                    

                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    
                    logit = model(data)
                    loss = loss_fn(logit, label)
                    loss_mean = loss.mean()

                    opt.zero_grad()
                    loss_mean.backward()
                    opt.step()

                    acc_mean = t.mean((logit.max(1)[1]==label).float().detach().cpu())
                    logs[self.curr_epoch]['train_acc'].update(acc_mean.detach().cpu(),len(label))
                    logs[self.curr_epoch]['train_loss'].update(loss_mean.detach().cpu(), len(loss))
                logs = self.validation(model, loss_fn, sch, logs, model_name='Normal Training')

                if tune_save:
                    tune.report(loss= float(logs[self.curr_epoch]['valid_loss'].avg), accuracy = float(logs[self.curr_epoch]['valid_acc'].avg), test_accuracy = float(logs[self.curr_epoch]['test_acc'].avg), uuid = self.args.uuid)


                sch.step()
                
