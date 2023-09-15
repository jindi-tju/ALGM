import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MRF_layer(nn.Module):

    def __init__(self):
        super(MRF_layer, self).__init__()

        self.weight = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(1)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, pre, s, k):
        k_layer = k
        Q = pre
        for i in range(k_layer):
            Q1 = torch.log(pre + 1e-8)

            compatibility = (-1) * np.eye(2) + np.ones((2, 2))
            r = torch.FloatTensor(compatibility).cuda()

            Q2_A = s
            Q2_2 = Q @ r

            Q2 = F.softmax(-torch.matmul(Q2_A, Q2_2), dim=1)

            Q =F.softmax( Q1 * torch.sigmoid(self.weight) + Q2 * (1. - torch.sigmoid(self.weight)), dim=1)  #  这里还该不该归一化的问题 行和为1

        return Q