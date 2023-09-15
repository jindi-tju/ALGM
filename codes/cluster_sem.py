import os
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


INF = 1e20
from sklearn.cluster import KMeans



class Clustering_sk():
    def __init__(self, data, k, max_iter):
        self.data = data
        self.k = k
        self.max_iter = max_iter

    def Clustering_sk(self):
        data1 = self.data.detach().cpu().numpy()
        clf_KMeans = KMeans(n_clusters=self.k, max_iter=self.max_iter)
        c_pred = clf_KMeans.fit_predict(data1)   

        return c_pred

class SemGraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size = 32, epsilon=None, num_pers=16, metric_type='attention', device=None):
        super(SemGraphLearner, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.metric_type = metric_type
        self.cuda()
        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)])
            # print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            # print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.leakyrelu = nn.LeakyReLU(0.2)
            # print('[ GAT_Attention GraphLearner]')

        elif metric_type == 'tanh':
            self.input_size = input_size
            # print('[ tanh GraphLearner]')

        elif metric_type == 'cosine':
            pass

        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        # print('[ Graph Learner metric type: {} ]'.format(metric_type))

    def forward(self, context, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= len(self.linear_sims)
            attention = F.softmax(attention, dim=1)

            markoff_value = 0

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0

        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF

        elif self.metric_type == 'tanh':

            l = context.shape[0]
            weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(l, self.input_size, self.input_size)))
            w = weight.cuda()

            context_fc = torch.matmul(w, context.transpose(-1, -2))
            context_fc1 = context
            attention = torch.matmul(context_fc1, context_fc)

            a = attention.detach().cpu().numpy()
            for x in np.nditer(a, op_flags=['readwrite']):
                x[...] = math.tanh(x)

            attention = torch.from_numpy(a).cuda()
            markoff_value = 0

        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0

        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon:

            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        return attention

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix



