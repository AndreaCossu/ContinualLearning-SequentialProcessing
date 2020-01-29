import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size=None, relu=False,
        dropout=False, out_activation=None):
        '''
        :param nonlinear: if False last layer is linear, if True is nonlinear. Default False.
        :param output_size: None if output does not have to be computed. An integer otherwise. Default None.
        '''

        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.out_activation = out_activation

        self.linears = nn.ModuleList([nn.Linear(self.input_size, self.hidden_sizes[0], bias=True)])
        if self.dropout:
            self.dropouts = nn.ModuleList([nn.Dropout(p=0.5) for i in range(len(hidden_sizes)-1)])

        for i in range(1,len(self.hidden_sizes)):
            self.linears.append(nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i], bias=True))

        if self.output_size is not None:
            self.linears.append(nn.Linear(self.hidden_sizes[-1], self.output_size, bias=True))

        self.activation = torch.tanh if not relu else torch.relu

    def forward(self, x):
        '''
        :param x: (batch_size, n_features)

        :return h: last hidden layer activations (batch_size, hidden_size)
        :return out: output of the network (batch_size, output_size). None if output_size is None.
        '''

        h = self.linears[0](x)
        h = self.activation(h)

        for i in range(1,len(self.hidden_sizes)):
            h = self.linears[i](h)
            h = self.activation(h)
            if self.dropout:
                h = self.dropouts[i-1](h)


        if self.output_size is not None:
            out = self.linears[-1](h)
            if self.out_activation is not None:
                out = self.out_activation(out)
            return h, out
        else:
            return h, None
