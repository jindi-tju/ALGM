import numpy as np
import torch
from torch import nn

from multi_detection import MultiDetModule
from cluster_sem import Clustering_sk, SemGraphLearner
from mrf_layer import MRF_layer
from torch.autograd import Variable, Function
import torch.nn.functional as F


class ReverseLayerF(Function):

    @staticmethod
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x):
    return ReverseLayerF().apply(x)


class NNq(nn.Module):
    def __init__(self, opt, feature_dim=64+16+16, h_dim=64):
        super(NNq, self).__init__()

        self.opt = opt
        self.multiDetModule = MultiDetModule()
        self.semanticGraph = SemGraphLearner(96, self.opt['graph_learn_hidden_size'], self.opt['epsilon'],
                                             self.opt['num_pers'], self.opt['metric_type'])  # 不过mrfs时注释掉 .cuda()
        self.mrf = MRF_layer()

        ###Fake News Classifier
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h_dim, 2)
        )

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1',
                                          nn.Linear(2 * self.opt['event_hidden_size'], self.opt['event_hidden_size'], ))

        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.opt['event_hidden_size'], self.opt['n_clusters']))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        if opt['cuda']:
            self.cuda()

    def forward(self, text, image, test_flag):
        m = self.multiDetModule(text, image)

        # predit Event
        reverse_feature = grad_reverse(m)
        event_pre = self.domain_classifier(reverse_feature.cuda())

        # predit Fake News
        pre = self.classifier_corre(m)
        pre = F.softmax(pre, dim=1)

        s = self.semanticGraph(m)
    
        if self.opt['flag_mrf']:

            pre = self.mrf(pre, s, self.opt['K'])

        return pre, event_pre

