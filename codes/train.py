import os

import copy
import numpy as np
import pandas as pd
import random
import argparse
import torch
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from trainer import Trainer
from nnq_p import NNq
from data_loader import LoadData
from cluster_sem import Clustering_sk



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='twitter')
parser.add_argument('--save', type=str, default='/')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs.')
parser.add_argument('--draw', type=str, default='max', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--ratio',  type=float, default='1.0', help='train ratio.')
parser.add_argument('--flag_mrf', type=int, default=1, help='if use MRF layer.')
parser.add_argument('--flag_event', type=int, default=1, help='if use event.')
parser.add_argument('--K', type=int, default=5, help='the number of mrf layers.')
parser.add_argument('--n_clusters', type=int, default=200, help='the number of clusters.')
parser.add_argument('--metric_type', type=str, default='attention', help='Method for graph construction: attention, weighted_cosine, gat_attention, tanh')
parser.add_argument('--graph_learn_hidden_size', type=int, default=32, help='hidden size of graph learning')
parser.add_argument('--epsilon', type=float, default=0.1, help='graph sparsity epsilon.')
parser.add_argument('--num_pers', type=int, default=1, help='the number of heads in attention')
parser.add_argument('--clu_iters', type=int, default=10, help='clustering iteration number')
parser.add_argument('--epo_patience', type=int, default=15, help='early stop epoch patience')
parser.add_argument('--event_hidden_size', type=int, default=48, help='hidden size of graph learning')
parser.add_argument('--lambd', type=int, default=1, help='')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)
max_epoches = opt['epoch']

# ---  Load Data  ---
all_set = LoadData(opt['dataset'], opt['ratio'])
idx_train = all_set.train_idx


idx_test = all_set.test_idx
idx_all = list(range(all_set.l))

opt['num_node'] = all_set.l
opt['num_class'] = 2

target = torch.LongTensor(all_set.labels)


idx_train = torch.LongTensor(idx_train)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)

text = all_set.data_text
img = all_set.data_img


if opt['cuda']:
    target = target.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    text = text.cuda()
    img = img.cuda()


def stop_condition(epoch, patience, best_epo):
    """
    Checks have not exceeded max epochs and has not gone patience epochs without improvement.
    """
    now_epoch = epoch
    patience = patience
    best_epo = best_epo
    no_improvement = now_epoch >= best_epo + patience
    exceeded_max_epochs = now_epoch >= max_epoches
    return True if exceeded_max_epochs or no_improvement else False


nnq = NNq(opt)
trainer = Trainer(opt, nnq)

best1, best_f1 = 0.0, 0.0
best_p, best_r = 0.0, 0.0
best_epo = 0
b_classification_report = None
best_preds_test = np.zeros((idx_test.size(0), opt['num_class']))

if opt['flag_event']:
    e_target = Clustering_sk(img, opt['n_clusters'], opt['clu_iters']).Clustering_sk()
else:
    e_target = np.zeros(idx_all.size(0))

e_target = torch.LongTensor(e_target)
if opt['cuda']:
    e_target = e_target.cuda()

loop = tqdm(range(max_epoches))


for epoch in loop:

    loss, accuracy_train = trainer.update(text, img, idx_train, target, e_target, opt['flag_mrf'], 1,
                                          opt['flag_event'])

    accuracy_test, test_loss1, classification_report\
            = trainer.predit(text, img, idx_test, target, opt['flag_mrf'], 0)


    if accuracy_test > best1:
        best_test_loss1 = test_loss1
        best1 = accuracy_test
        best_epo = epoch
        b_classification_report = classification_report

        state = dict([('model', copy.deepcopy(trainer.model.state_dict())),
                          ('optim', copy.deepcopy(trainer.optimizer.state_dict()))])

    loop.set_postfix(now_acc_train=accuracy_train, now_acc_test=accuracy_test)
    loop.set_description(f'Epoch[{epoch}/{max_epoches}]')

    print(f"Epoch[{epoch},train_loss: {loss:.4f},test_loss: {test_loss1:.4f},now_acc_train:{accuracy_train:.4f},"
          f"  now_acc_test:{accuracy_test:.4f},"
          f"  best_acc_test: {best1:.4f},{best_epo}")

    print("now_classification_report\n", classification_report)

    if stop_condition(epoch, opt['epo_patience'], best_epo) or (epoch == (max_epoches-1)):
        break

print(f"TEST BEST-----best_epoch:{best_epo},acc: {best1:.5f}")
print("TEST BEST------classification_report\n", b_classification_report)


trainer.model.load_state_dict(state['model'])
trainer.optimizer.load_state_dict(state['optim'])

if opt['save'] != '/':
    trainer.save(opt['save'] + '/best.pt')



