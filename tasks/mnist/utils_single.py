import torch
import torch.nn.functional as F


def permute(x, permutation):
    '''
    :param x: (B, 28*28, 1)
    :param permutation: (28*28)
    '''
    return torch.gather( x, 1, permutation.unsqueeze(0).repeat(x.size(0),1).unsqueeze(2) ) 

def permute_cifar(x, permutation):
    '''
    :param x: (B, 32*32, 3)
    :param permutation: (32*32)
    '''
    return torch.gather( x, 1, permutation.unsqueeze(0).repeat(x.size(0),1).unsqueeze(2).repeat(1,1,x.size(2)) )

    
def accuracy(output, target):
    '''
    A metric to evaluate results
    '''

    probs = torch.nn.functional.softmax(output, dim=1)
    winners = probs.argmax(dim=1)

    acc = (winners == target).sum().float() / target.size(0)

    return acc.item()
    


def train_autoencoder(ae, optimizer, x, reconstruction_loss, device):
    idx_ae = torch.tensor(list(range(x.size(1)-1,-1,-1)), device=device).long() # reverse sequence order
    ae.train()

    optimizer.zero_grad()

    h_ae = ae.reset_hidden(batch_size=x.size(0))
    out, _ = ae(x,h_ae)

    #re = reconstruction_loss(out,x[:,idx_ae,:]) # revert sequence using idx_ae
    re = reconstruction_loss(out,x[:,:,:])

    re.backward()

    optimizer.step()

    return re.item()

def test_autoencoder(train_autoencoders, x, reconstruction_loss, device):

    with torch.no_grad():
        autoencoders = train_autoencoders[0]

        reconstruction_errors = []
        idx_ae = torch.tensor(list(range(x.size(1)-1,-1,-1)), device=device).long() # reverse sequence order
        for ae in autoencoders:
            ae.eval()
            h_ae = ae.reset_hidden(batch_size=x.size(0))
            out, _ = ae(x[:,:,:],h_ae)
            #re = reconstruction_loss(out,x[:,idx_ae,:]) # revert sequence use idx_ae
            re = reconstruction_loss(out,x[:,:,:])

            reconstruction_errors.append(re.item())

        module_id = reconstruction_errors.index(min(reconstruction_errors))

        return reconstruction_errors, module_id




def train(train_models, modelname, x,y, accuracy, device, output_size, max_grad_norm=5.0):
    model = train_models[modelname][0]
    optimizer = train_models[modelname][1]
    criterion = train_models[modelname][2]

    model.train()

    optimizer.zero_grad()

    if modelname=='lstm' or modelname=='rnn' or modelname=='alstm':

        h = model.reset_memory_state()

        h, predictions = model(x, h)


    else: # lmn, almn


        h = model.reset_memory_state()

        predictions = torch.empty(x.size(0), x.size(1), output_size, requires_grad=False, device=device).float()


        for i in range(x.size(1)):
            h, predictions[:, i, :] = model(x[:, i, :], h)
    

    predictions = predictions[:, -1, :]

    loss = criterion(predictions, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()


    with torch.no_grad():
        acc = accuracy(predictions, y)


    return loss.item(), acc


def test(train_models, modelname, x,y, accuracy, device, output_size, module_id=None):

    with torch.no_grad():
        model = train_models[modelname][0]

        model.eval()


        if modelname=='lstm' or modelname=='rnn' or modelname=='alstm':

            if modelname == 'alstm':
                h = model.reset_memory_state(batch_size=x.size(0), module_id=module_id)
                h, predictions = model(x, h, task_id=module_id)
            else:
                h = model.reset_memory_state(batch_size=x.size(0))
                h, predictions = model(x, h)


        else: # lmn, almn

            predictions = torch.empty(x.size(0), x.size(1), output_size, requires_grad=False, device=device).float()

            if modelname=='almn':
                h = model.reset_memory_state(batch_size=x.size(0), module_id=module_id)
                for i in range(x.size(1)):
                    h, predictions[:, i, :] = model(x[:, i, :], h, task_id=module_id)
            else:
                h = model.reset_memory_state(batch_size=x.size(0))
                for i in range(x.size(1)):
                    h, predictions[:, i, :] = model(x[:, i, :], h)


        predictions = predictions[:, -1, :]

        loss = F.cross_entropy(predictions, y, reduction='mean')

        acc = accuracy(predictions, y)

        return loss.item(), acc




def train_ewc(ewc, task_id, train_models, modelname, x, y, acc_f, device, output_size, max_grad_norm=5.0):

    model = train_models[modelname][0]
    optimizer = train_models[modelname][1]
    criterion = train_models[modelname][2]

    model.train()

    optimizer.zero_grad()

    if modelname=='lstm' or modelname=='rnn':

        h = model.reset_memory_state()

        h, predictions = model(x, h)

    else: 

        h = model.reset_memory_state()

        predictions = torch.empty(x.size(0), x.size(1), output_size, requires_grad=False, device=device).float()


        for i in range(x.size(1)):
            h, predictions[:, i, :] = model(x[:, i, :], h)

    predictions = predictions[:, -1, :]
    
    loss = criterion(predictions, y)

    loss_out = loss.item()

    loss += ewc.ewc_penalty(model, modelname, task_id)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    with torch.no_grad():
        acc = acc_f(predictions, y)


    return loss_out, acc

def accumulate_backward(train_models, modelname, x_data, y_data, device, output_size, max_grad_norm=5.0):

    model = train_models[modelname][0]
    optimizer = train_models[modelname][1]
    criterion = train_models[modelname][2]

    model.train()

    optimizer.zero_grad()


    for x,y in zip(x_data, y_data):

        if modelname=='lstm' or modelname=='rnn':

            h = model.reset_memory_state()

            h, predictions = model(x, h)

        else: # lmn

            h = model.reset_memory_state()

            predictions = torch.empty(x.size(0), x.size(1), output_size, requires_grad=False, device=device).float()


            for i in range(x.size(1)):
                h, predictions[:, i, :] = model(x[:, i, :], h)

        loss = criterion(predictions[:, -1, :], y)

        loss.backward()