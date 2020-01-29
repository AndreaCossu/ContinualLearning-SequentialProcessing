import torch
import torch.nn as nn
from core.MLP import MLP

class Memory(nn.Module):

    def __init__(self, memory_size, functional_size, output_size=None, out_activation=None):
        '''
        :param functional_size: size of the hidden activations of the functional component.
        :param output_size: None to use LMN-A, an integer to use LMN-B. Default None.
        '''

        super(Memory, self).__init__()

        self.memory_size = memory_size
        self.functional_size = functional_size
        self.output_size = output_size
        self.out_activation = out_activation

        self.linear_memory = nn.Linear(self.memory_size, self.memory_size, bias=True)
        self.linear_functional = nn.Linear(self.functional_size, self.memory_size, bias=True)

        if self.output_size is not None:
            self.output_layer = nn.Linear(self.memory_size, self.output_size, bias=True)

    def forward(self, f_h, m_h):
        '''
        :param f_h: (batch_size, memory_size + functional_size) activations of functional component concatenated with previous memory hidden state.

        :return h: memory hidden state (batch_size, memory_size)
        :return out: output of the network (batch_size, output_size). None if output_size is None.
        '''

        self.m = self.linear_memory(m_h)

        self.f = self.linear_functional(f_h)
        h = self.m + self.f

        if self.output_size is not None:
            out = self.output_layer(h)
            if self.out_activation is not None:
                out = self.out_activation(out)
            return h, out
        else:
            return h, None

class LMN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, memory_size, device,
                batch_size, type_A=True, out_activation=None, orthogonal=True):
        '''
        To use a LMN with a single layer functional component pass as hidden_sizes a list of one element e.g.: [10]
        In case of multi layer functional component only the last layer is used to compute the memory state.

        :param hidden_sizes: a list containing hidden sizes for the layers of the functional component
        :param type_A: True to use LMN-A, False to use LMN-B. Default True.
        '''

        super(LMN, self).__init__()

        self.memory_size = memory_size
        self.input_size_f = input_size + self.memory_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.type_A = type_A
        self.device = device
        self.batch_size = batch_size

        self.current_h_f = None

        if self.type_A:
            self.functional = MLP(self.input_size_f, self.hidden_sizes, self.output_size,
                out_activation=out_activation).to(self.device)
            self.memory = Memory(self.memory_size, self.hidden_sizes[-1]).to(self.device)
        else:
            self.functional = MLP(self.input_size_f, self.hidden_sizes).to(self.device)
            self.memory = Memory(self.memory_size, self.hidden_sizes[-1],
                self.output_size, out_activation=out_activation).to(self.device)

        if orthogonal:
            nn.init.orthogonal_(self.memory.linear_memory.weight)

    def forward(self, x, h_m):
        '''
        :param x: (batch_size, input_size) current input
        :param h_m: (batch_size, memory_size) hidden state of the memory

        :return h_m_new: memory hidden state
        :return out: (batch_size, output_size) output of the LMN for current input x
        '''

        f_input = torch.cat((x,h_m), dim=1)
        if self.type_A:
            self.current_h_f, out = self.functional(f_input)
            h_m_new, _ = self.memory(self.current_h_f, h_m)
        else:
            self.current_h_f, _ = self.functional(f_input)
            h_m_new, out = self.memory(self.current_h_f, h_m)

        return h_m_new, out


    def reset_memory_state(self, batch_size=None):

        b = self.batch_size if batch_size is None else batch_size

        return torch.zeros(b, self.memory_size, requires_grad=True, device=self.device)
