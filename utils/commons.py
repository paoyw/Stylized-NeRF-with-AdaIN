import os
import random
import json
import numpy as np
import torch

def same_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_checkpath(args):
    os.makedirs(args.checkpath, exist_ok=args.overwrite) 
    with open(os.path.join(args.checkpath, 'args.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=2))

    args.log_file = os.path.join(args.checkpath, args.log_file)
    with open(args.log_file, 'w') as f:
        f.write(json.dumps([], indent=2))
