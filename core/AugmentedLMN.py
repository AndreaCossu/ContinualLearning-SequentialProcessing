import torch
import torch.nn as nn
from core.MLP import MLP
import os

class MemoryModule(nn.Module):
    def __init__(self, memory_size, functional_size, prev_mem_size=0, output_size=None, out_activation=None):
        '''
        :param memory_size: size of the memory hidden state
        :param functional_size: size of the hidden activations of the functional component.
        :param prev_mem_size: memory size of previous module or 0 to not connect it.
        :param output_size: None to use LMN-A, an integer to use LMN-B. Default None.
        :param out_activation: output activation function or None to not use it
        '''

        super(MemoryModule, self).__init__()

        self.memory_size = memory_size
        self.functional_size = functional_size
        self.output_size = output_size
        self.out_activation = out_activation
        self.input_size_m = self.memory_size + prev_mem_size

        self.linear_memory = nn.Linear(self.input_size_m, self.memory_size, bias=True)
        self.linear_functional = nn.Linear(self.functional_size, self.memory_size, bias=True)

        if self.output_size is not None:
            self.output_layer = nn.Linear(self.memory_size, self.output_size, bias=True)

    def forward(self, f_h, f_m):
        '''
        :param f_h: (batch_size, memory_size + functional_size) activations of functional component concatenated with previous memory hidden state.
        :param f_m previous memory hidden state

        :return h: memory hidden state (batch_size, memory_size)
        :return out: output of the network (batch_size, output_size). None if output_size is None.
        '''

        self.m = self.linear_memory(f_m)
        self.f = self.linear_functional(f_h)

        h = self.m + self.f

        if self.output_size is not None:
            out = self.output_layer(h)
            if self.out_activation is not None:
                out = self.out_activation(out)
            return h, out
        else:
            return h, None



class LMNModule(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, memory_size,
        previous_memory_size, previous_hidden_size, device, batch_size, feed_mem=False,
        type_A=True, out_activation=None, orthogonal=False):

        '''
        To use a LMN with a single layer functional component pass as hidden_sizes a list of one element e.g.: [10]
        In case of multi layer functional component only the last layer is used to compute the memory state.

        :param hidden_sizes: a list containing hidden sizes for the layers of the functional component
        :param type_A: True to use LMN-A, False to use LMN-B. Default True.
        :feed_mem: True if previous memory has to be fed to the current memory. False otherwise. Default True.
        '''

        super(LMNModule, self).__init__()

        self.memory_size = memory_size
        self.previous_hidden_size = previous_hidden_size
        self.previous_memory_size = previous_memory_size
        self.input_size_f = input_size + self.memory_size + self.previous_hidden_size + self.previous_memory_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.type_A = type_A
        self.device = device
        self.batch_size = batch_size
        self.current_h_f = None
        self.out_activation = out_activation

        self.mem_connection = 0 if feed_mem is False else self.previous_memory_size
        if self.type_A:
            self.functional = MLP(self.input_size_f, self.hidden_sizes, self.output_size, out_activation=self.out_activation).to(self.device)
            self.memory = MemoryModule(self.memory_size, self.hidden_sizes[-1],
            prev_mem_size=self.mem_connection).to(self.device)
        else:
            self.functional = MLP(self.input_size_f, self.hidden_sizes).to(self.device)
            self.memory = MemoryModule(self.memory_size, self.hidden_sizes[-1],
            prev_mem_size=self.mem_connection, output_size=self.output_size, out_activation=self.out_activation).to(self.device)

        if orthogonal:
            nn.init.orthogonal_(self.memory.linear_memory.weight)

    def forward(self, x, h_m, prev_h_f=None, prev_h_m=None):
        '''
        :param x: (batch_size, input_size) current input
        :param h_m: (batch_size, memory_size) hidden state of the memory

        :return h_m_new: memory hidden state
        :return out: (batch_size, output_size) output of the LMN for current input x
        '''

        if prev_h_f is not None and prev_h_m is not None:
            f_input = torch.cat((x,h_m, prev_h_f, prev_h_m), dim=1)
        else:
            f_input = torch.cat((x, h_m), dim=1)

        if self.type_A:
            self.current_h_f, out = self.functional(f_input)
            if self.mem_connection > 0:
                h_m = torch.cat((h_m, prev_h_m), dim=1)

            h_m_new, _ = self.memory(self.current_h_f, h_m)
        else:
            self.current_h_f, _ = self.functional(f_input)
            if self.mem_connection > 0:
                h_m = torch.cat((h_m, prev_h_m), dim=1)

            h_m_new, out = self.memory(self.current_h_f, h_m)


        return h_m_new, out


    def reset_memory_state(self, batch_size=None):

        b = self.batch_size if batch_size is None else batch_size

        return torch.zeros(b, self.memory_size, requires_grad=True, device=self.device)



class AugmentedLMN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, memory_size, device,
        batch_size, type_A=True, feed_mem=False, out_activation=None, orthogonal=True):

        super(AugmentedLMN, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = [hidden_sizes]
        self.output_sizes = [output_size]
        self.memory_sizes = [memory_size]
        self.device = device
        self.batch_size = batch_size
        self.type_A = type_A
        self.feed_mem = feed_mem
        self.out_activation = out_activation
        self.orthogonal = orthogonal

        lmn = LMNModule(input_size, hidden_sizes, output_size, memory_size, 0, 0,
            device, self.batch_size, feed_mem, type_A, out_activation=self.out_activation,
            orthogonal=orthogonal)

        self.lmns = nn.ModuleList([lmn])

    def add_new_module(self, optimizer, hidden_sizes=None, output_size=None, memory_size=None, type_A=None, batch_size=None):

        for param in self.lmns[-1].parameters():
            param.requires_grad = False

        previous_hidden_size = self.hidden_sizes[-1][-1]
        previous_memory_size = self.memory_sizes[-1]

        hidden_sizes = hidden_sizes if hidden_sizes is not None else self.hidden_sizes[-1]
        output_size = output_size if output_size is not None else self.output_sizes[-1]
        memory_size = memory_size if memory_size is not None else self.memory_sizes[-1]
        type_A = type_A if type_A is not None else self.type_A
        batch_size = batch_size if batch_size is not None else self.batch_size

        new_almn = LMNModule(self.input_size, hidden_sizes, output_size, memory_size, previous_memory_size,
            previous_hidden_size, self.device,  batch_size, self.feed_mem, type_A,
            out_activation=self.out_activation, orthogonal=self.orthogonal).to(self.device)

        self.lmns.append(new_almn)

        del optimizer.param_groups[0]
        optimizer.add_param_group({'params': [p for p in new_almn.parameters()]})

        self.hidden_sizes.append(hidden_sizes)
        self.output_sizes.append(output_size)
        self.memory_sizes.append(memory_size)


    def forward(self, x, h_m, task_id=None):
        '''
        :param task_id: id of the input task (start from 0)

        :return new_hs: list of memory states for each module
        :return outs: list of outputs for each module
        '''


        new_hs = []
        h_m_new, out = self.lmns[0](x, h_m[0])
        new_hs.append(h_m_new)

        n_modules = len(self.lmns)
        if n_modules > 1:
            id = n_modules if task_id is None else task_id+1

            for mod in range(1,id):
                previous_h_f = self.lmns[mod-1].current_h_f
                h_m_new, out = self.lmns[mod](x, h_m[mod], previous_h_f, h_m_new)
                new_hs.append(h_m_new)

        return new_hs, out


    def reset_memory_state(self, module_id=None, batch_size=None):

        id = len(self.lmns) if module_id is None else module_id+1
        hs = []
        for i in range(id):
            h = self.lmns[i].reset_memory_state(batch_size)
            hs.append(h)

        return hs


    def save_augmented(self, path):
        save_lmns = {}
        for i, lmn in enumerate(self.lmns):
            save_lmns[i] = lmn.state_dict()

        torch.save(save_lmns, os.path.join(path, 'almn.pt'))

    def load_augmented(self, path, device):
        check = torch.load(os.path.join(path,'almn.pt'), map_location=device)

        self.lmns = nn.ModuleList([])

        for i, lmn in check.items():
            if i==0:
                load_lmn = LMNModule(self.input_size, self.hidden_sizes[-1], self.output_sizes[-1], self.memory_sizes[-1], 0, 0,
                            device, self.batch_size, self.feed_mem, self.type_A, out_activation=self.out_activation,
                            orthogonal=self.orthogonal).to(device)
            else:
                load_lmn =  LMNModule(self.input_size, self.hidden_sizes[-1], self.output_sizes[-1],
                    self.memory_sizes[-1], self.memory_sizes[-1],
                    self.hidden_sizes[-1][-1], device, self.batch_size, self.feed_mem, self.type_A,
                    out_activation=self.out_activation, orthogonal=self.orthogonal).to(device)


            load_lmn.load_state_dict(lmn)
            self.lmns.append(load_lmn)
