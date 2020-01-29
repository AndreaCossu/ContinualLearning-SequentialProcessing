import torch
import torch.nn as nn

class BaselineRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, device, lstm=False, batch_size=1,
                num_layers=1, dropout=0., bidirectional=False, relu=False, orthogonal=False):
        '''
        Baseline recurrent network.

        :param lstm: False to choose a RNN or True to choose a LSTM. Default False.

        '''

        super(BaselineRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation = 'relu' if relu else 'tanh'
        self.dropout = dropout if self.num_layers > 1 else 0.
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.batch_size = batch_size
        self.device = device
        self.lstm = lstm
        self.orthogonal = orthogonal

        if not self.lstm:
            self.rnn_module = nn.RNN(self.input_size, self.hidden_size,
                num_layers=self.num_layers, nonlinearity=self.activation,
                batch_first=True, dropout=self.dropout,
                bidirectional=self.bidirectional).to(self.device)
            if self.orthogonal:
                for _, hh, _, _ in self.rnn_module.all_weights:
                    nn.init.orthogonal_(hh)
        else:
            self.rnn_module = nn.LSTM(self.input_size, self.hidden_size,
                self.num_layers, batch_first=True, dropout=self.dropout,
                bidirectional=self.bidirectional).to(self.device)
            if self.orthogonal:
                for _, hh, _, _ in self.rnn_module.all_weights:
                    # lstm divides hidden matrix into 4 chunks
                    # https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
                    for j in range(0, hh.size(0), self.hidden_size): 
                        nn.init.orthogonal_(hh[j:j+self.hidden_size])

        self.linear = nn.Linear(self.directions*self.hidden_size, self.output_size).to(self.device)



    def forward(self, x, h):
        '''
        :param x: (batch_size, seq_len, input_size)
        :param h: hidden state of the recurrent module

        :return out: (batch_size, seq_len, directions*hidden_size)
        :return h: hidden state of the recurrent module
        '''

        out, h = self.rnn_module(x, h)

        out = self.linear(out)

        return h, out


    def reset_memory_state(self, batch_size=None):
        '''
        :param batch_size: size of current batch. If None default batch size used. Default None.
        '''

        b = self.batch_size if batch_size is None else batch_size
        if self.lstm:
            h = (
                torch.zeros(self.directions*self.num_layers, b, self.hidden_size,
                    device=self.device, requires_grad=True),

                torch.zeros(self.directions*self.num_layers, b, self.hidden_size,
                    device=self.device, requires_grad=True)
            )
        else:
            h = torch.zeros(self.directions*self.num_layers, b, self.hidden_size,
                                device=self.device, requires_grad=True)

        return h
