# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:58:24 2019

@author: Administrator
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


__all__ = ['AngularSoftmaxWithLoss']


class AngularSoftmaxWithLoss(nn.Module):
    """"""
    def __init__(self, gamma=0):
        super(AngularSoftmaxWithLoss, self).__init__()
        self.gamma = gamma
        self.iter = 0
        self.lambda_min = 5.0
        self.lambda_max = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.iter += 1
#        print(target)
#        target = target.view(-1, 1)
#        print(target)

#        index = input[0].data * 0.0
#        print(index)
#        print(len(index))
#        index.scatter_(1, target.unsqueeze(1).data.cpu(), 1)
#        print(index)

        targets = torch.zeros(input.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1)
        print(targets)
        index = targets.to(torch.device('cuda'))
        index = index.byte()


        # Tricks
        # output(θyi) = (lambda * cos(θyi) + (-1) ** k * cos(m * θyi) - 2 * k)) / (1 + lambda)
        #             = cos(θyi) - cos(θyi) / (1 + lambda) + Phi(θyi) / (1 + lambda)
        self.lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.iter))
        output = input * 1.0
        print(output)
        output[index] -= input[0][index] * 1.0 / (1 + self.lamb)
        output[index] += input[1][index] * 1.0 / (1 + self.lamb)

        # softmax loss
        logit = F.log_softmax(output)
        logit = logit.gather(1, target).view(-1)
        pt = logit.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss