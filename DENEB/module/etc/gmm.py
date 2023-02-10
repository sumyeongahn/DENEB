import numpy as np
import scipy.stats as stats
import torch as t

from sklearn.mixture import GaussianMixture as GMM

import numpy as np
import pandas as pd
from tqdm import tqdm





def get_singular_vector(features, labels):
    '''
    To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
    features: hidden feature vectors of data (numpy)
    labels: correspoding label list
    '''
    
    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            _, _, v = np.linalg.svd(features[labels==index])
            singular_vector_dict[index] = v[0]
            pbar.update(1)

    return singular_vector_dict


def get_score(singular_vector_dict, features, labels, normalization=True):
    '''
    Calculate the score providing the degree of showing whether the data is clean or not.
    '''
    if normalization:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat/np.linalg.norm(feat))) for indx, feat in enumerate(tqdm(features))]
    else:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat)) for indx, feat in enumerate(tqdm(features))]
        
    return np.array(scores)


def fit_mixture(scores, labels, p_threshold=0.5):
    '''
    Assume the distribution of scores: bimodal gaussian mixture model
    
    return clean labels
    that belongs to the clean cluster by fitting the score distribution to GMM
    '''
    
    clean_labels = []
    unlabel_labels = []
    indexes = np.array(range(len(scores)))
    probs = np.zeros(len(scores))


    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)

        # gmm = GMM(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)
        gmm = GMM(n_components=2, max_iter = 20, reg_covar = 5e-4, tol=1e-2)
        
        feats_[np.isnan(feats_)] = 0
        gmm.fit(feats_)
        prob = gmm.predict_proba(feats_)
        prob = prob[:,gmm.means_.argmax()]
        probs[cls_index] = prob

    return t.tensor(probs)
    