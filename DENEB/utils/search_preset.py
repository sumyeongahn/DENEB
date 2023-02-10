from ray import tune

index = 0

def get_preset(alg, dataset): 
    
    if dataset == 'colored_mnist':
        config = {
        "lr": tune.choice([0.01]),
        "batch_size": tune.choice([256]),
        "lr_decay_rate": tune.choice([0.1]),
        "lr_decay_opt": tune.choice([30])
        }
    elif dataset == 'corrupted_cifar':
        config = {
        "lr": tune.choice([0.001]),
        "batch_size": tune.choice([256]),
        "lr_decay_rate": tune.choice([0.1]),
        "lr_decay_opt": tune.choice([20])
        }

    if  'gce' in alg:
        config['gce_q'] = tune.choice([0.1, 0.3, 0.5, 0.7, 0.9])
    
    elif 'sce' in alg:
        config['alpha'] = tune.choice([0.5, 0.7])
        config['beta'] = tune.choice([0.2, 0.7])

    elif 'elr' in alg:
        config['lbd'] = tune.choice([1, 3, 5, 7])
        config['beta'] = tune.choice([0.7, 0.9])
        config['warmup'] = tune.choice([0, 10, 30])
        config['ema_alpha'] = tune.choice([0.9, 0.99, 0.997])
        config['ema_step'] = tune.choice([30000, 40000, 50000])
        config['mixup_alpha'] = tune.choice([0.5, 1.0, 2.0, 5.0])
        config['coef_step'] = tune.choice([0, 100, 1000])
        

    elif  'coteaching+' in alg:
        config['num_gradual'] = tune.choice([1, 5, 10, 20])
        config['forget_rate'] = tune.choice(['opt'])
        config['warmup'] = tune.choice([10, 20])

    elif  'coteaching' in alg:
        config['num_gradual'] = tune.choice([1, 5, 10, 20])
        config['forget_rate'] = tune.choice(['opt'])

    elif 'aum' in alg:
        config['perc'] = tune.choice([1, 10, 30, 50, 99])
    
    elif 'fdividemix' in alg:
        config['alpha'] = tune.choice([0.5, 1.0, 2.0, 5.0])
        config['lambda_u'] = tune.choice([0, 0.5, 1, 15, 25])
        config['p_threshold'] = tune.choice([0.1, 0.3, 0.5, 0.7])
        config['warmup'] = tune.choice([0, 5, 10])
        config['T'] = tune.choice([0.1, 0.5])
    
    elif 'dividemix' in alg:
        config['alpha'] = tune.choice([0.5, 1.0, 2.0, 5.0])
        config['lambda_u'] = tune.choice([0, 0.5, 1, 15, 25])
        config['p_threshold'] = tune.choice([0.1, 0.3, 0.5, 0.7])
        config['warmup'] = tune.choice([0, 5, 10])
        config['T'] = tune.choice([0.1, 0.5])
    
    
    if 'lff' in alg:
        config['m_alpha'] = tune.choice([0.1, 0.5, 0.7, 0.9])
        config['q'] = tune.choice([0.1, 0.3, 0.5, 0.7, 0.9])

    elif 'disen' in alg:
        config['m_alpha'] = tune.choice([0.1, 0.5, 0.7, 0.9])
        config['q'] = tune.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        config['warmup'] = tune.choice([0, 10, 20])
        config['lambda_dis'] = tune.choice([1, 10, 30])
        config['lambda_swap'] = tune.choice([1, 10, 30])

    elif 'jtt' in alg:
        config['bias_epochs'] = tune.choice([10, 20, 50, 100])
        config['lbd_up'] = tune.choice([30., 60., 100])

    elif 'deneb' in alg:
        config['p_threshold'] = tune.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        config['warmup'] = tune.choice([0, 5, 10])
        config['tau'] = tune.choice([0.5, 1.0, 5.0, 10.0, 20.0])
    
    return config


def get_preset_dir(config, uuids):
    config['uuid'] = tune.choice([f'{str(uuid)}' for uuid in uuids]) 
    return config