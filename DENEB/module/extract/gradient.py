import torch as t
import torch.nn.functional as F
from tqdm import tqdm


def module_type(name):
    return ''.join([i for i in name if not i.isdigit()])

class grad_fn:
    def __init__(self, name, data_len):
        self.name = name
        self.data = None
        self.data_len = data_len
        self.curr = 0
        self.norm = True

    def save_grad(self,grad):
        if self.data == None:
            grad = grad.reshape([1,-1]).detach().cpu()
            self.data_size = grad.shape[-1]
            self.data = t.zeros(self.data_len, self.data_size)
            self.data[self.curr] = grad
            self.curr += 1
        else:
            grad = grad.reshape([1,-1]).detach().cpu()
            self.data[self.curr] = grad
            self.curr += 1
                

class grad_ext:
    def __init__(self, model, target_layer, data_len):
        self.model = model
        self.target_layer = target_layer
        self.save_list = {}
        self.hook_list = {}
        self.data_len = data_len
        self.initialize()
        
    def initialize(self):
        for name, module in self.model._modules.items():
            if module_type(name) in self.target_layer:
                curr_name = name
                self.save_list[curr_name] = grad_fn(curr_name,self.data_len)
                self.hook_list[curr_name] = module.weight.register_hook(self.save_list[curr_name].save_grad)
            else:
                for  name_m, module_m in module._modules.items():
                    if module_type(name_m) in self.target_layer:
                        curr_name = name + '_' + name_m
                        self.save_list[curr_name] = grad_fn(curr_name,self.data_len)
                        self.hook_list[curr_name] = module_m.weight.register_hook(self.save_list[curr_name].save_grad)
                    else:
                        for  name_b, module_b in module_m._modules.items():
                            if module_type(name_b) in self.target_layer:
                                curr_name = name + '_' + name_m + '_' + name_b
                                self.save_list[curr_name] = grad_fn(curr_name,self.data_len)
                                self.hook_list[curr_name] = module_b.weight.register_hook(self.save_list[curr_name].save_grad)
                            else:
                                for  name_l, module_l in module_b._modules.items():
                                    if module_type(name_l) in self.target_layer:
                                        curr_name = name + '_' + name_m + '_' + name_b + '_' + name_l
                                        self.save_list[curr_name] = grad_fn(curr_name,self.data_len)
                                        self.hook_list[curr_name] = module_l.weight.register_hook(self.save_list[curr_name].save_grad)
                                    else:
                                        for  name_f, module_f in module_l._modules.items():
                                            if module_type(name_f) in self.target_layer:
                                                curr_name = name + '_' + name_m + '_' + name_b + '_' + name_l + '_' + name_f
                                                self.save_list[curr_name] = grad_fn(curr_name,self.data_len)
                                                self.hook_list[curr_name] = module_f.weight.register_hook(self.save_list[curr_name].save_grad)
    
    def remove(self):
        for name in self.hook_list.keys():
            self.hook_list[name].remove()
            
    def __call__(self, x):
        x = self.model(x)
        # for name, module in self.model._modules.items():
            
        #     print(x.shape)
        #     if name  == 'fc':
        #         x = x.reshape(len(x), -1)
        #     x = module(x)
        return x


def estimate_grads(loader, model, loss_fn, opt, args, label_type='given'):
    model.train()
    all_grads = []
    all_idx = []

    grad = grad_ext(model, ['fc'], len(loader.dataset))

    with tqdm(loader) as progress:
        for batch_idx, data_tuple in enumerate(progress):
            data = data_tuple[0].to(args.device)
            idx = data_tuple[4]
            
            output = grad(data)
            if label_type == 'uniform':
                label = (t.ones_like(output)/output.shape[1]).to(args.device)
            elif label_type == 'softmax':
                label = F.softmax(output,dim=1).to(args.device)
            else:
                label = data_tuple[1].to(args.device)
                
            loss = loss_fn(output, label)

            for sample_idx in range(len(label)):
                opt.zero_grad()
                loss[sample_idx].backward(retain_graph= True)
            
            if batch_idx == 0:
                all_idx = idx
            else:
                all_idx = t.cat([all_idx, idx], dim=0)
            
    sort = t.argsort(all_idx.reshape(-1))
    
    for name in grad.save_list.keys():
        if 'fc' in name:
            all_grads = grad.save_list[name].data
    
    grad.remove()

    return all_grads[sort]
