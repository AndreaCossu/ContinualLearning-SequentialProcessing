import random
import torch
import torch.nn.functional as F
from tasks.audioset.preprocess import generator_audioset, select_category
from collections import defaultdict


'''
label_conversion = {}
target = 0
for el in filtered_eval:
    label_conversion[el] = target
    target = (target + 1) % 10
'''
label_conversion = {4: 0, 6: 1, 10: 2, 525: 3, 14: 4, 15: 5, 17: 6, 18: 7, 21: 8, 23: 9, 
24: 0, 25: 1, 33: 2, 34: 3, 38: 4, 43: 5, 45: 6, 49: 7, 53: 8, 54: 9,
56: 0, 58: 1, 59: 2, 60: 3, 62: 4, 63: 5, 67: 6, 70: 7, 77: 8, 79: 9,
82: 0, 92: 1, 94: 2, 96: 3, 105: 4, 149: 5, 174: 6, 182: 7, 183: 8, 184: 9,
198: 0, 201: 1, 203: 2, 207: 3, 209: 4, 211: 5, 215: 6, 292: 7, 309: 8, 312: 9,
317: 0, 318: 1, 327: 2, 339: 3, 342: 4, 346: 5, 347: 6, 348: 7, 361: 8, 367: 9,
368: 0, 369: 1, 377: 2, 382: 3, 386: 4, 387: 5, 390: 6, 391: 7, 398: 8, 400: 9,
403: 0, 407: 1, 411: 2, 414: 3, 415: 4, 419: 5, 421: 6, 442: 7, 450: 8, 452: 9,
453: 0, 454: 1, 467: 2, 468: 3, 469: 4, 470: 5, 471: 6, 481: 7, 483: 8, 493: 9, 
498: 0, 500: 1}

path_save_autoencoders = 'saved_models/audioset/'

def load_unbal_data(filename, tasks, block_size):
    gen = generator_audioset(filename, block_size=block_size)

    xs = defaultdict(list)
    ys = defaultdict(list)

    for x, y, _ in gen:
        for i, task in enumerate(tasks):
            idx = select_category(task, y, one_category=True)
            if idx is not None:
                x_task, y_task = x[idx], y[idx]
                y_task = transform_labels(y_task, module=False)
                xs[i].append(x_task)
                ys[i].append(y_task)


    for k in xs.keys():
        xs[k] = torch.cat(xs[k])
        ys[k] = torch.cat(ys[k])

    return xs, ys


def split_audioset(x,y,test_size=0.2):
    random_idx = list(range(x.size(0)))
    random.shuffle(random_idx)
    x_shuffled, y_shuffled = x[random_idx], y[random_idx]
    end = int(x.size(0) * test_size)
    x_train, x_val, y_train, y_val = x_shuffled[end:], x_shuffled[:end], y_shuffled[end:], y_shuffled[:end]

    return x_train, x_val, y_train, y_val

def random_segments(x,y, batch_size):
    random_idx = list(range(x.size(0)))
    random.shuffle(random_idx)
    random_idx = random_idx[:batch_size]

    return x[random_idx], y[random_idx]

def accuracy(output, target):
    probs = torch.nn.functional.softmax(output, dim=1)
    winners = probs.argmax(dim=1)

    acc = (winners == target).sum().float() / target.size(0)

    return acc.item()

def transform_labels(y, module=True, formatted=False):
    '''
    y: (B, n_classes)

    It assumes one 1 for each row
    '''

    if not formatted:
        y = y.nonzero()[:,1].long()

    if not module:
        return y
        
    # convert labels in [0, 9]
    for i in range(y.size(0)):
        y[i] = label_conversion[ y[i].item() ]

    return y

def train(train_models, modelname, x, y, acc_f, device, output_size, max_grad_norm=5.0):

    x = x.to(device)
    y = y.to(device)

    model = train_models[modelname][0]
    optimizer = train_models[modelname][1]
    criterion = train_models[modelname][2]

    model.train()

    optimizer.zero_grad()

    if modelname=='lstm' or modelname=='rnn' or modelname=='alstm' or modelname=='lstm_sep':

        h = model.reset_memory_state()

        h, predictions = model(x, h)

        predictions = predictions[:, -1, :]

        loss = criterion(predictions, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

    else: # lmn, almn, lmn_sep

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
        acc = acc_f(predictions, y)


    return loss.item(), acc


def test(train_models, modelname, x,y, acc_f, device, output_size, module_id=None):

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        model = train_models[modelname][0]

        model.eval()

        if modelname=='lstm' or modelname=='rnn' or modelname=='alstm' or modelname=='lstm_sep':

            if modelname == 'alstm':
                h = model.reset_memory_state(batch_size=x.size(0), module_id=module_id)
                h, predictions = model(x, h, task_id=module_id)
            else:
                h = model.reset_memory_state(batch_size=x.size(0))
                h, predictions = model(x, h)

        else: # lmn, almn, lmn_sep

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

        acc = acc_f(predictions, y)

        return acc, loss.item()

def train_autoencoder(ae, optimizer, x, reconstruction_loss, device):

    x = x.to(device)

    ae.train()

    optimizer.zero_grad()

    h_ae = ae.reset_hidden(batch_size=x.size(0))
    out, _ = ae(x,h_ae)

    re = reconstruction_loss(out,x)

    re.backward()

    optimizer.step()

    return re.item()

def test_autoencoder(train_autoencoders, x, reconstruction_loss, device, small_version):

    x = x.to(device)

    with torch.no_grad():

        autoencoders = train_autoencoders[0]

        reconstruction_errors = []

        for id, ae in enumerate(autoencoders):

            if not small_version and id == 0:
                reconstruction_errors.append(1000)
                continue

            ae.eval()
            h_ae = ae.reset_hidden(batch_size=x.size(0))
            out, _ = ae(x,h_ae)
            re = reconstruction_loss(out,x)

            reconstruction_errors.append(re.item())

        module_id = reconstruction_errors.index(min(reconstruction_errors))

    return reconstruction_errors, module_id


def train_ewc(ewc, cat_id_p, train_models, modelname, x,y, acc_f, device, output_size, max_grad_norm=5.0):
    x = x.to(device)
    y = y.to(device)

    model = train_models[modelname][0]
    optimizer = train_models[modelname][1]
    criterion = train_models[modelname][2]

    model.train()

    optimizer.zero_grad()

    if modelname=='lstm' or modelname=='rnn' or modelname=='alstm' or modelname=='lstm_sep':

        h = model.reset_memory_state()

        h, predictions = model(x, h)


    else: # lmn, almn, lmn_sep

        h = model.reset_memory_state()

        predictions = torch.empty(x.size(0), x.size(1), output_size, requires_grad=False, device=device).float()

        for i in range(x.size(1)):
            h, predictions[:, i, :] = model(x[:, i, :], h)

    predictions = predictions[:, -1, :]

    loss = criterion(predictions, y)

    loss_out = loss.item()

    loss += ewc.ewc_penalty(model, modelname, cat_id_p)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    with torch.no_grad():
        acc = acc_f(predictions, y)


    return loss_out, acc

def accumulate_backward(train_models, modelname, x, y, device, output_size, max_grad_norm=5.0):

    model = train_models[modelname][0]
    optimizer = train_models[modelname][1]
    criterion = train_models[modelname][2]

    model.train()

    optimizer.zero_grad()

    h = model.reset_memory_state(batch_size=x.size(0))

    if modelname=='lstm' or modelname=='rnn' or modelname=='alstm' or modelname=='lstm_sep':


        h, predictions = model(x, h)


    else: # lmn, almn, lmn_sep

        predictions = torch.empty(x.size(0), x.size(1), output_size, requires_grad=False, device=device).float()

        for i in range(x.size(1)):
            h, predictions[:, i, :] = model(x[:, i, :], h)

    predictions = predictions[:, -1, :]

    loss = criterion(predictions, y)

    loss.backward()