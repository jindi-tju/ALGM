import os
import numpy as np
import torch
from torch import nn
from multi_detection import MultiDetModule
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report



def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class Trainer(object):
    def __init__(self, opt, model):
        self.crosmodel = MultiDetModule().cuda()
        self.opt = opt
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        if opt['cuda']:
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        self.model.reset()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])


    def update(self, text, img, idx_train, target, event_target, mrf_flag, test_flag, event_flag):  # target 这里直接只传的train的部分 now这个和下面的预测是数据过两次版
        print("\nin train!\n")
        print("\nlen_train_id:", len(idx_train))

        if self.opt['cuda']:
            text = text.cuda()
            img = img.cuda()
            idx_train = idx_train.cuda()
            target = target.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits, event_logits= self.model(text, img, test_flag)


        floss = self.criterion(logits[idx_train], target[idx_train])
        if event_flag:
            eloss = self.criterion(event_logits, event_target)
            loss = floss + eloss
        else:
            loss = floss

        loss.backward()
        self.optimizer.step()
        
        preds_train = torch.max(logits[idx_train], dim=1)[1]
        correct_train = preds_train.eq(target[idx_train]).double()
        accuracy_train = correct_train.sum() / idx_train.size(0)

        return loss.item(), accuracy_train.item()


    def predit(self, text, img, idx_test, target, mrf_flag, test_flag):
        if self.opt['cuda']:
            text = text.cuda()
            img = img.cuda()
            idx_test = idx_test.cuda()
            target = target.cuda()

        self.model.eval()
        with torch.no_grad():

            logits, event_logits = self.model(text, img, test_flag)

            preds_test = torch.max(logits[idx_test], dim=1)[1]
            correct_test = preds_test.eq(target[idx_test]).double()
            accuracy_test = correct_test.sum() / len(idx_test)

            loss = self.criterion(logits[idx_test], target[idx_test])

            y_true = target[idx_test].detach().cpu().numpy()
            y_pre = preds_test.detach().cpu().numpy()

            target_names = ['real 0', 'fake 1']

        return accuracy_test.item(),loss.item(),  classification_report(y_true, y_pre, target_names=target_names, digits=5)




    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])

