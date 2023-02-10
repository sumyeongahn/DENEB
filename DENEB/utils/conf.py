import yaml

def inject(args, data):
    for key in data.keys():
        if 'config/' in key:
            _key = key.split('config/')[1]
            args.__dict__[_key] = data[key]
        elif 'uuid' in key:
            args.__dict__[key] = data[key]
    return args


def get_unknown(args, unknown):
    if len(unknown) == 0:
        args.option = 'none'
    else:
        args.option = ''
    for idx in range(len(unknown)):
        if '--' in unknown[idx]:
            key = unknown[idx][2:]
            args.__dict__[key] = str(unknown[idx+1])
            print(key, str(unknown[idx+1]))
            args.option += f'{key}_{unknown[idx+1]}_'
    args.option = args.option[:-1]
    
    return args

def get_search(args, option):
    args.option = ''
    for key in option.keys():
        args.__dict__[key] = option[key]
        args.option += f'{key}_{option[key]}_'
    args.option = args.option[:-1]
    return args