import os
import copy
import torch, gc
from collections import defaultdict, OrderedDict
import numpy as np

gc.collect()
torch.cuda.empty_cache()

opt = dict()

opt['dataset'] = 'twitter'
opt['input_dropout'] = 0.5
opt['dropout'] = 0.5   # 0
opt['optimizer'] = 'adam'
opt['lr'] = 1e-4
opt['decay'] = 2e-1
opt['self_link_weight'] = 1.0
opt['epoch'] = 200
opt['ratio'] = 1
opt['flag_mrf'] = 1
opt['flag_event'] = 1
opt['K'] = 1
opt['n_clusters'] = 20
opt['metric_type'] = 'weighted_cosine'
opt['graph_learn_hidden_size'] = 48
opt['event_hidden_size'] = 48
opt['epsilon'] = 0.4
opt['num_pers'] = 1
opt['clu_iters'] = 10
opt['epo_patience'] = 50



def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd


def run(opt):
    opt_ = copy.deepcopy(opt)
    print("opt\n",opt_)
    os.system(generate_command(opt_))

for k in range(1):
    seed = k + 2
    opt['seed'] = seed
    run(opt)
