import torch
import torch.nn as nn
import torch.optim as optim


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, batch_size):

        super(LSTMAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.batch_size = batch_size

        self.encoder = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True).to(device)
        self.decoder = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True).to(device)

        self.linear = nn.Linear(self.hidden_dim, self.input_dim).to(device)

    def forward(self, input, h_enc):
        '''
        :param input: (B, L, I)
        and hidden states initialized from autoencoder

        :return out: (B, L, I)parser.add_argument('--hidden_sizes_lmn', nargs='+', type=int, default=[128], help='layers of functional component of LMN')

        and hidden state to forward to autoencoder
        '''

        out, h_enc = self.encoder(input, h_enc)
        
        # TODO: use [x_t ; h_t^enc] as input at each timestep 
        out, _ = self.decoder(torch.zeros(input.size(0), input.size(1), self.hidden_dim, device=self.device), h_enc)
        
        # TODO: linear layer should flatten output and then reshape it again.
        out = self.linear(out)

        return out, h_enc


    def reset_hidden(self, batch_size=None):

        b = batch_size if batch_size is not None else self.batch_size
        h_enc = (
            torch.zeros(1, b, self.hidden_dim,
                device=self.device, requires_grad=True),

            torch.zeros(1, b, self.hidden_dim,
                device=self.device, requires_grad=True)
        )
        return h_enc


if __name__ == "__main__":

    inp_size = 10
    batch_size = 1

    autoencoder = LSTMAutoencoder(input_dim=inp_size, hidden_dim=100, device='cpu', batch_size=batch_size)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(autoencoder.parameters())

    x = torch.randn(batch_size, 5, inp_size)

    for i in range(100):
        optimizer.zero_grad()

        h_enc = autoencoder.reset_hidden()
        out, h_enc = autoencoder(x, h_enc)


        loss = criterion(out, x)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        #print((out-x)**2)
        print(torch.norm((out-x)**2))
