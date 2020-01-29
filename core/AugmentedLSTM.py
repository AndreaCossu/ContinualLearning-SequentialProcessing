import torch
import torch.nn as nn
import os


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, prev_hidden_size, device, batch_size=1, orthogonal=False):

        super(LSTMModule, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size+prev_hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        if orthogonal:
            for _, hh, _, _ in self.lstm.all_weights:
                # lstm divides hidden matrix into 4 chunks
                # https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
                for j in range(0, hh.size(0), self.hidden_size): 
                    nn.init.orthogonal_(hh[j:j+self.hidden_size])



    def forward(self, x, h, prev_h=None):

        input_f = x

        if prev_h is not None:
            input_f = torch.cat((x, prev_h), dim=2) # (B, L, I+H)

        out_h, h = self.lstm(input_f, h)
        out = self.linear(out_h)

        return h, out, out_h

    def reset_memory_state(self, batch_size=None):
        b = batch_size if batch_size is not None else self.batch_size

        h = (
            torch.zeros(1, b, self.hidden_size,
                device=self.device, requires_grad=True),

            torch.zeros(1, b, self.hidden_size,
                device=self.device, requires_grad=True)
        )

        return h

class AugmentedLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, device, batch_size=1, orthogonal=False):

        super(AugmentedLSTM, self).__init__()


        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_sizes = [hidden_size]
        self.orthogonal = orthogonal

        lstm0 = LSTMModule(input_size, hidden_size, output_size, 0, device, batch_size, orthogonal=orthogonal).to(device)

        self.lstms = nn.ModuleList([lstm0])

    def add_new_module(self, optimizer, hidden_size=None, output_size=None, batch_size=None):

        for param in self.lstms[-1].parameters():
            param.requires_grad = False


        hidden_size = hidden_size if hidden_size is not None else self.hidden_sizes[-1]
        output_size = output_size if output_size is not None else self.output_size
        batch_size = batch_size if batch_size is not None else self.batch_size
        prev_hidden_size = self.hidden_sizes[-1]

        new_lstm = LSTMModule(self.input_size, hidden_size, output_size,
            prev_hidden_size, self.device,  batch_size, orthogonal=self.orthogonal).to(self.device)

        self.lstms.append(new_lstm)

        del optimizer.param_groups[0]
        optimizer.add_param_group({'params': [p for p in new_lstm.parameters()]})

        self.hidden_sizes.append(hidden_size)

    def forward(self, x, h, task_id=None):

        new_hs = []
        h_new, out, out_h = self.lstms[0](x, h[0])
        new_hs.append(h_new)

        n_modules = len(self.lstms)
        if n_modules > 1:
            id = n_modules if task_id is None else task_id+1

            for mod in range(1,id):
                h_new, out, out_h = self.lstms[mod](x, h[mod], out_h)
                new_hs.append(h_new)

        return new_hs, out


    def reset_memory_state(self, module_id=None, batch_size=None):

        id = len(self.lstms) if module_id is None else module_id+1
        hs = []
        for i in range(id):
            h = self.lstms[i].reset_memory_state(batch_size)
            hs.append(h)

        return hs

    def save_augmented(self, path):
        save_lstms = {}
        for i, lstm in enumerate(self.lstms):
            save_lstms[i] = lstm.state_dict()

        torch.save(save_lstms, os.path.join(path, 'alstm.pt'))

    def load_augmented(self, path, device):
        check = torch.load(os.path.join(path,'alstm.pt'), map_location=device)

        self.lstms = nn.ModuleList([])

        for i, lstm in check.items():
            if i==0:
                load_lstm = LSTMModule(self.input_size, self.hidden_sizes[-1], \
                                            self.output_size,\
                                            0, device, self.batch_size).to(device)
            else:
                load_lstm = LSTMModule(self.input_size, self.hidden_sizes[-1], self.output_size, \
                            self.hidden_sizes[-1], device, self.batch_size).to(device)

            load_lstm.load_state_dict(lstm)
            self.lstms.append(load_lstm)
