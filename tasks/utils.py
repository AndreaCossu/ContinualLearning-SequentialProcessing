import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from functools import reduce
from torch.nn.modules.loss import _Loss


save_test_results = 'test_perf.txt'
path_save_models = 'saved_models'

class OrthogonalLoss(_Loss):
    '''
    Loss that adds to the classic BCE a penalty term
    in order to have orthogonal memory matrix
    '''

    def __init__(self, criterion, memory, device, lamb=0.1, beta=0., model='lmn'):
        
        '''
        :param model: string representing the model
        :param criterion: main loss to be used
        :param memory: the Memory class of the (A)LMN currently in use
        :param lamb: hyperparameter to control orthogonalization
        :param beta: hyperparameter to control hidden state norm
        '''

        super(OrthogonalLoss, self).__init__(None, None, 'mean')

        self.device = device
        self.model = model
        self.memory = memory
        self.lamb = lamb
        self.beta = beta
        self.criterion = criterion

    def update_memory(self, memory):
        '''
        Update memory, in case new modules are added to the ALMN or ALSTM
        '''

        self.memory = memory

    def forward(self, input, target, hs=None):
        '''
        :param hs: (B, 2*L-1, H) hidden states for each batch and for each sequence item
        '''

        loss = self.criterion(input, target)


        if hs is not None:
            batch_losses = loss.mean(dim=2).mean(dim=1)
            T = hs.size(1)
            h_norms = torch.norm(hs, dim=2, p=2) # (B, 2*L-1) norms

            norms_sum = (h_norms[:, 1:] - h_norms[:, :-1])**2

            norms_penalty = self.beta * (torch.sum(norms_sum, dim=1) / float(T))

        if self.model.endswith('lmn'):
            orth_penalty = self.lamb * torch.norm(
                torch.matmul(self.memory.linear_memory.weight, self.memory.linear_memory.weight.transpose(0,1)) -
                torch.eye(self.memory.linear_memory.weight.size(0) , device=self.device)
                )**2
        elif self.model == 'rnn' or self.model == 'lstm' or self.model == 'alstm':
            orth_penalty = self.lamb * torch.norm(
                torch.matmul(self.memory.all_weights[0][1], self.memory.all_weights[0][1].transpose(0,1)) -
                torch.eye(self.memory.all_weights[0][1].size(0) , device=self.device)
                )**2   

        if hs is None:
            total_loss = loss.mean() +  orth_penalty
        else:
            total_loss = (batch_losses + norms_penalty).mean() + orth_penalty

        return total_loss



def monitor_orthogonality(args, writer, task_id, epoch, train_models):
    with torch.no_grad():
        for model in args.models:

            if model == 'almn':
                memory_matrix = train_models[model][0].lmns[-1].memory.linear_memory.weight
            elif model == 'lmn':
                memory_matrix = train_models[model][0].memory.linear_memory.weight
            elif model == 'rnn' or model == 'lstm':
                memory_matrix = train_models[model][0].rnn_module.all_weights[0][1]
            elif model == 'alstm':
                memory_matrix = train_models[model][0].lstms[-1].lstm.all_weights[0][1]

            visualize = torch.matmul(memory_matrix.transpose(0,1), memory_matrix).cpu()
            writer.add_image(model+"_memory/"+str(task_id), visualize, epoch, dataformats='HW')



def save_model(model, modelname, path):
    if modelname == 'alstm' or modelname == 'almn':
        model.save_augmented(path)
    else:
        torch.save(model.state_dict(), os.path.join(path, modelname+'.pt'))

def load_models(model, modelname, path, device):
    check = torch.load(os.path.join(path,modelname+'.pt'), map_location=device)

    if modelname == 'alstm' or modelname == 'almn':
        model[0].load_augmented(path, device)
    else:
        model[0].load_state_dict(check)

    model[0].eval()

    return model

def save_autoencoders(autoencoders, path):

    for i,ae in enumerate(autoencoders):
        torch.save(ae.state_dict(), os.path.join(path,'ae_'+str(i)+'.pt'))

def load_autoencoders(autoencoders, device, path):
    for i in range(len(autoencoders)):
        check = torch.load(os.path.join(path,'ae_'+str(i)+'.pt'), map_location=device)
        autoencoders[i].load_state_dict(check)
        autoencoders[i].eval()

    return autoencoders

def MSEMasked(input, target, masks=None):

    input = torch.sigmoid(input)
    
    pointwise_loss = (input - target)**2

    if masks is not None:
        pointwise_loss = pointwise_loss * masks
        loss = torch.sum(pointwise_loss) / float(torch.nonzero(masks).size(0))
    else:
        loss = torch.sum(pointwise_loss) / float(reduce( lambda a,b: a*b, list(target.size()) ))
        
    return loss


def monitor(writer, value, name, t):
    '''
    Write to tensorboardX a single scalar value
    '''
    writer.add_scalar(name, value.item(), t)


def configure_plots(folder):
    '''
    Set plot folder to folder by creating it if it does not exist.
    '''

    default="plots/"

    if not os.path.isdir(os.path.join(folder, path_save_models)):
        try:
            os.makedirs(os.path.join(folder, path_save_models))
        except OSError:
            print("Error when creating experiment folder")
            folder = default

    if folder[-1] != '/':
        folder += '/'

    return folder

def write_test_results(plot_folder, accs, losses, res=None, module_acc=None):

    with open(os.path.join(plot_folder, 'loss_'+save_test_results), 'w') as f:
        for model, loss in losses.items():
            for t, l in enumerate(loss):
                f.write(str(t+1) + ',' + model + ',' + str(l) + '\n')

    with open(os.path.join(plot_folder, 'acc_'+save_test_results), 'w') as f:
        for model, acc in accs.items():
            for t, a in enumerate(acc):
                f.write(str(t+1) + ',' + model + ',' + str(a) + '\n')

    if res is not None:
        with open(os.path.join(plot_folder, 'autoencoders_'+save_test_results), 'w') as f:
            for t, re in enumerate(res):
                for ae_id, ae_re in enumerate(re):
                    f.write(str(t+1) + ',' + str(ae_id+1) + ',' + str(ae_re) + '\n')

    if module_acc is not None:
        np.savetxt(os.path.join(plot_folder, 'module_acc_'+save_test_results), module_acc, delimiter=',', fmt='%.2f') 

def write_intermediate_test_results(plot_folder, accs, losses):
    with open(os.path.join(plot_folder, 'intermediate_loss_'+save_test_results), 'w') as f:
        for model, loss in losses.items():
            for t, l in loss.items():
                f.write(str(t) + ',' + model + ',' + str(l) + '\n')

    with open(os.path.join(plot_folder, 'intermediate_acc_'+save_test_results), 'w') as f:
        for model, acc in accs.items():
            for t, a in acc.items():
                f.write(str(t) + ',' + model + ',' + str(a) + '\n')


def write_configuration(args, dest):
    '''
    Write the input argument passed to the script to a file
    '''

    args_d = vars(args)
    with open(os.path.join(dest, 'conf.txt'), 'w') as f:
        for k, v in args_d.items():
            f.write(str(k) + ": " + str(v)+"\n")

def print_num_parameters(args, train_models):
    '''
    Print the number of parameters of the models
    '''

    for model in args.models:
        print(model , ': ', len(list(train_models[model][0].parameters())))
        par = 0
        for param in train_models[model][0].parameters():
            curr = 1
            for i in range(len(param.size())):
                curr *= param.size(i)
            par += curr
        print(par)



def get_colors(n):
    return plt.cm.get_cmap('hsv', n)

def plot(args, v1, model, type, v2=None):

    tasks, plot_folder, epochs = args[0], args[1], args[2]

    colors = get_colors(len(v1)*3)

    plt.figure()
    plt.title(model+'_'+type+'-'+str(epochs))
    for task in range(len(v1)):
        task_vals = v1[task]
        plt.plot(range(len(task_vals)), task_vals, color=colors(task*3), label=model+'-'+str(tasks[task]))
        if v2 is not None:
            test_task_vals = v2[task]
            plt.plot( range(len(test_task_vals)), test_task_vals, ls='--', color=colors(task*3))

    plt.legend(loc='best')


    plt.savefig(os.path.join(plot_folder, model+'_'+type+'.png'))

