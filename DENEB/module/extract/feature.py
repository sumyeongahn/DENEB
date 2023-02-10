from tqdm import tqdm
import numpy as np
import torch as t
    
def get_features(model, dataloader):
    '''
    Concatenate the hidden features and corresponding labels 
    '''
    labels = np.empty((0,))

    model.eval()
    model.cuda()
    
    for batch_idx, data_tuple in enumerate(dataloader):
        data = data_tuple[0]
        label = data_tuple[1]
        idx = data_tuple[4]
        data, label = data.cuda(), label.long()
        feature = model.extract(data).detach().cpu().numpy()

        labels = np.concatenate((labels, label.cpu()))
        if batch_idx == 0:
            features = feature
            all_idx = idx
        else:
            features = np.concatenate((features, feature), axis=0)
            all_idx = t.cat([all_idx, idx],dim=0)
    argsort = t.argsort(all_idx)
    return features[argsort].squeeze(), labels[argsort]