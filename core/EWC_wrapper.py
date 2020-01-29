'''
Inspired by ContinualAI notebook.
https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb
'''


import torch
import torch.nn as nn

class EWC(nn.Module):
    def __init__(self, device, lamb=1):
        '''

        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for EWC.
        '''

        super(EWC, self).__init__()

        self.lamb = lamb
        self.device = device

        self.saved_params = {}
        self.fisher = {}

    def ewc_penalty(self, model, modelname, current_task_id):
        '''
        :param model: model to be optimized
        :param modelname: name of the model
        :param current_task_id: current task ID.
        '''

        total_penalty = torch.tensor(0, dtype=torch.float32).to(self.device)

        # for each previous task (if any)
        for task in range(current_task_id):
            for param, saved_param, fisher in zip(model.parameters(), self.saved_params[modelname][task], self.fisher[modelname][task]):
                total_penalty += self.lamb * ( fisher * (param - saved_param).pow(2) ).sum()

        return total_penalty

    def compute_fisher(self, model, modelname, current_task_id, x=None, y=None, criterion=None):
        '''
        :param model: model to be optimized
        :param modelname: name of the model
        :param current_task_id: current task ID.
        :param x: training data (iterable of minibatches). If x,y, and
                    criterion are not provided, it is assumed that
                    gradient has been already accumulated over training set.
                    Otherwise forward pass + gradient computation is performed.
        :param y: training labels (iterable of minibatches). If None see x.
        :param criterion: loss function. If None see x.
        '''


        if x is not None and y is not None and criterion is not None:
            for x_batch, y_batch in zip(x,y):
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()

        if modelname not in self.saved_params.keys():
            self.saved_params[modelname] = {}
        if modelname not in self.fisher.keys():
            self.fisher[modelname] = {}

        # store learned parameters and fisher coefficients
        # no need to store all the tensor metadata, just its data (data.clone())
        self.saved_params[modelname][current_task_id] = [ param.data.clone() for param in model.parameters() ]
        self.fisher[modelname][current_task_id] = [ param.grad.data.clone().pow(2) for param in model.parameters() ]
