import torch
import os
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from core.BaselineRNN import BaselineRNN
from core.LMN import LMN
from core.AugmentedLMN import AugmentedLMN
from core.AugmentedLSTM import AugmentedLSTM
# from core.LSTMAutoencoder import LSTMAutoencoder
from core.LSTMAutoencoder2 import LSTMAutoencoder
from torch.nn.modules.loss import _Loss

class CL_Experiment():
    '''
    A manager of CL experiments
    '''

    def __init__(self, args, save_test_results='test_perf.txt', path_save_models='saved_models'):
        '''
        Create CL experiment manager
        '''

        self.args = args

        self.save_test_results = save_test_results
        self.path_save_models = path_save_models

        self.device = torch.device('cpu')


    def get_device(self):
        '''
        Choose device: cpu or cuda
        '''

        mode = 'cpu'
        if self.args.cuda:
            if torch.cuda.is_available():
                print('Using ', torch.cuda.device_count() ,' GPU(s)')
                mode = 'cuda'
            else:
                print("WARNING: No GPU found. Using CPUs...")
        else:
            print('Using 0 GPUs')

        self.device = torch.device(mode)

        return self.device


    def configure_plots(self):
        '''
        Set plot folder by creating it if it does not exist.
        '''

        default="plots"

        if not os.path.isdir(os.path.join(self.args.plot_folder, self.path_save_models)):
            try:
                os.makedirs(os.path.join(self.args.plot_folder, self.path_save_models))
            except OSError:
                print("Error when creating experiment folder")
                self.args.plot_folder = default

        # if folder[-1] != '/':
        #     folder += '/'

        return self.args.plot_folder
    

    def get_writer(self):
        '''
        Get Tensorboard writer and store on tensorboard directory
        '''

        writer = SummaryWriter(os.path.join(self.args.plot_folder,'tensorboard'))

        return writer 

    
    def save_autoencoders(self, autoencoders, path):
        for i,ae in enumerate(autoencoders):
            torch.save(ae.state_dict(), os.path.join(path, 'ae_'+str(i)+'.pt'))

    def load_autoencoders(self, autoencoders, path):
        for i in range(len(autoencoders)):
            check = torch.load(os.path.join(path, 'ae_'+str(i)+'.pt'), map_location=self.device)
            autoencoders[i].load_state_dict(check)
            autoencoders[i].eval()

        return autoencoders
    
    def save_model(self, model, modelname, path):
        if modelname == 'alstm' or modelname == 'almn':
            model.save_augmented(path)
        else:
            torch.save(model.state_dict(), os.path.join(path, modelname+'.pt'))

    def load_models(self, model, modelname, path):
        check = torch.load(os.path.join(path,modelname+'.pt'), map_location=self.device)

        if modelname == 'alstm' or modelname == 'almn':
            model[0].load_augmented(path, self.device)
        else:
            model[0].load_state_dict(check)

        model[0].eval()

        return model

    def create_models(self):
        '''
        Create models for CL experiment.
        '''

        train_models = defaultdict(list)
        train_autoencoders = []

        if 'alstm' in self.args.models or 'almn' in self.args.models:
            train_autoencoders.append([LSTMAutoencoder(self.args.input_size, self.args.hidden_size_autoencoder, self.device, self.args.batch_size).to(self.device) for i in range(len(self.args.tasks))])
            train_autoencoders.append([torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.decay_ae) for ae in train_autoencoders[0]])

        if 'rnn' in self.args.models:
            train_models['rnn'].append(BaselineRNN(self.args.input_size, self.args.hidden_size_rnn, self.args.output_size, self.device,
                batch_size=self.args.batch_size, lstm=False, num_layers=self.args.layers_rnn, orthogonal=self.args.orthogonal))
            train_models['rnn'].append(torch.optim.Adam(train_models['rnn'][0].parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay))
            if self.args.orthogonal_loss:
                train_models['rnn'].append(OrthogonalLoss(torch.nn.CrossEntropyLoss(reduction='mean'), train_models['rnn'][0].rnn_module, self.device, self.args.lamb, 0, 'rnn'))
            else:
                train_models['rnn'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

        if 'lstm' in self.args.models:
            train_models['lstm'].append(BaselineRNN(self.args.input_size, self.args.hidden_size_rnn, self.args.output_size, self.device,
                batch_size=self.args.batch_size, lstm=True, num_layers=self.args.layers_rnn, orthogonal=self.args.orthogonal))
            train_models['lstm'].append(torch.optim.Adam(train_models['lstm'][0].parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay))
            if self.args.orthogonal_loss:
                train_models['lstm'].append(OrthogonalLoss(torch.nn.CrossEntropyLoss(reduction='mean'), train_models['lstm'][0].rnn_module, self.device, self.args.lamb, 0, 'lstm'))
            else:
                train_models['lstm'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

        if 'alstm' in self.args.models:
            train_models['alstm'].append(AugmentedLSTM(self.args.input_size, self.args.hidden_size_rnn, self.args.output_size, self.device,
                batch_size=self.args.batch_size, orthogonal=self.args.orthogonal))
            train_models['alstm'].append(torch.optim.Adam(train_models['alstm'][0].parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay))
            if self.args.orthogonal_loss:
                train_models['alstm'].append(OrthogonalLoss(torch.nn.CrossEntropyLoss(reduction='mean'), train_models['alstm'][0].lstms[-1], self.device, self.args.lamb, 0, 'alstm'))
            else:
                train_models['alstm'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

        if 'lmn' in self.args.models:
            train_models['lmn'].append(LMN(self.args.input_size, self.args.hidden_sizes_lmn, self.args.output_size, self.args.memory_size_lmn,
                self.device, self.args.batch_size, type_A=self.args.type_A, orthogonal=self.args.orthogonal))
            train_models['lmn'].append(torch.optim.Adam(train_models['lmn'][0].parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay))
            if self.args.orthogonal_loss:
                train_models['lmn'].append(OrthogonalLoss(torch.nn.CrossEntropyLoss(reduction='mean'), train_models['lmn'][0].memory, self.device, self.args.lamb, 0))
            else:
                train_models['lmn'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

        if 'almn' in self.args.models:
            train_models['almn'].append(AugmentedLMN(self.args.input_size, self.args.hidden_sizes_lmn, self.args.output_size,
                self.args.memory_size_lmn, self.device, self.args.batch_size, type_A=self.args.type_A, feed_mem=self.args.feed_mem, orthogonal=self.args.orthogonal))
            train_models['almn'].append(torch.optim.Adam(train_models['almn'][0].parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay))
            if self.args.orthogonal_loss:
                train_models['almn'].append(OrthogonalLoss(torch.nn.CrossEntropyLoss(reduction='mean'), train_models['almn'][0].lmns[0].memory, self.device, self.args.lamb, 0))
            else:
                train_models['almn'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

        if self.args.load:
            if 'alstm' in self.args.models or 'almn' in self.args.models:
                train_autoencoders[0] = self.load_autoencoders(train_autoencoders[0], os.path.join(self.args.plot_folder, self.path_save_models))
            for model in self.args.models:
                train_models[model] = self.load_models(train_models[model], model, os.path.join(self.args.plot_folder, self.path_save_models))

        return train_models, train_autoencoders



#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################



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
