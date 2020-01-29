import torch
import argparse
import pickle
import os
import numpy as np

from core.EWC_wrapper import EWC
from core.BaselineRNN import BaselineRNN
from core.LMN import LMN
from core.AugmentedLMN import AugmentedLMN
from core.AugmentedLSTM import AugmentedLSTM
from core.LSTMAutoencoder import LSTMAutoencoder
from tasks.audioset.utils import *
from tasks.utils import *
from collections import defaultdict
from tasks.audioset.preprocess import load_all_data, select_category, filtered_eval

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, nargs='+', default=[50000, 6000], help='epochs to train.')
parser.add_argument('--hidden_size_rnn', type=int, default=64, help='units of RNN')
parser.add_argument('--layers_rnn', type=int, default=1, help='layers of RNN')
parser.add_argument('--models', nargs='+', type=str, default=['almn'], help='models to train: lstm, alstm, rnn, lmn, almn')
parser.add_argument('--bidirectional', action="store_true", help="use bidirectional LSTM")
parser.add_argument('--hidden_sizes_lmn', nargs='+', type=int, default=[64], help='layers of functional component of LMN')
parser.add_argument('--memory_size_lmn', type=int, default=64, help='memory size of LMN')
parser.add_argument('--type_A', action="store_true", help='choose LMN-A ')
parser.add_argument('--feed_mem', action="store_true", help='feed previous memory module to current memory module')
parser.add_argument('--threshold_acc', type=float, default=1.01, help='add new ALMN module if test accuracy is below this threshold')

parser.add_argument('--separate_modules', action="store_true", help='train also a separate lmn and lstm for each task')

parser.add_argument('--ewc_lambda', type=float, default=0.0, help='Train with EWC.')


parser.add_argument('--hidden_size_autoencoder', type=int, default=40, help='hidden size of the autoencoders')
parser.add_argument('--lr_ae', type=float, default=1e-4, help='optimizer hyperparameter')
parser.add_argument('--decay_ae', type=float, default=1e-3, help='optimizer hyperparameter')

# optimizer parameters
parser.add_argument('--weight_decay', type=float, default=1e-3, help='optimizer hyperparameter')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer hyperparameter')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer hyperparameter')
parser.add_argument('--batch_size', type=int, default=3, help='batch size')

parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Value to clip gradient norm.')

parser.add_argument('--not_test', action="store_true", help='disable final test')
parser.add_argument('--unbalanced', action="store_true", help='use also unbalanced data')

parser.add_argument('--not_intermediate_test', action="store_true", help='Test intermediate results.')

parser.add_argument('--save', action="store_true", help='save models')
parser.add_argument('--load', action="store_true", help='load models')

parser.add_argument('--cuda', action="store_true", help='use gpu')
parser.add_argument('--print_every', type=int, default=100, help='print information every print_every steps')
parser.add_argument('--plot_folder', type=str, default='plots/swc/', help='folder in which to put saved plots. Created if not existing.')

################################### Init params and folders
args = parser.parse_args()


mode = 'cpu'
if args.cuda:
    if torch.cuda.is_available():
        print('Using ', torch.cuda.device_count() ,' GPU(s)')
        mode = 'cuda'
    else:
        print("WARNING: No GPU found. Using CPUs...")
else:
    print('Using 0 GPUs')


plot_folder = configure_plots(args.plot_folder)


device = torch.device(mode)

################################### Create models

if args.ewc_lambda > 0:
    ewc = EWC(device, lamb=args.ewc_lambda)

input_size = 128
output_size = 10
input_ae = input_size

train_models = defaultdict(list)
train_autoencoders = []

task_losses = defaultdict(list)
task_acc = defaultdict(list)
task_acc_val = defaultdict(list)
task_l_val = defaultdict(list)

task_re = []

if len(args.epochs) == 1:
    small_version = True
else:
    small_version = False

if small_version:
    tasks = [ # indexes of categories list
        filtered_eval[0:10],
        filtered_eval[60:70],
        filtered_eval[70:80],
        filtered_eval[80:90],
    ]
else:
    tasks = [ # indexes of categories list
        filtered_eval[0:50],
        filtered_eval[60:70],
        filtered_eval[70:80],
        filtered_eval[80:90],
    ]


if 'alstm' in args.models or 'almn' in args.models:
    train_autoencoders.append([LSTMAutoencoder(input_ae, args.hidden_size_autoencoder, device, args.batch_size).to(device) for i in range(len(tasks))])
    train_autoencoders.append([torch.optim.Adam(ae.parameters(), lr=args.lr_ae, weight_decay=args.decay_ae) for ae in train_autoencoders[0]])

if 'rnn' in args.models:
    train_models['rnn'].append(BaselineRNN(input_size, args.hidden_size_rnn, output_size, device,
        batch_size=args.batch_size, lstm=False, num_layers=args.layers_rnn))
    train_models['rnn'].append(torch.optim.RMSprop(train_models['rnn'][0].parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, momentum=args.momentum))
    train_models['rnn'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

if 'lstm' in args.models:
    train_models['lstm'].append(BaselineRNN(input_size, args.hidden_size_rnn, output_size, device,
        batch_size=args.batch_size, lstm=True, num_layers=args.layers_rnn, bidirectional=args.bidirectional))
    train_models['lstm'].append(torch.optim.RMSprop(train_models['lstm'][0].parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, momentum=args.momentum))
    train_models['lstm'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

if 'alstm' in args.models:
    train_models['alstm'].append(AugmentedLSTM(input_size, args.hidden_size_rnn, output_size, device,
        batch_size=args.batch_size))
    train_models['alstm'].append(torch.optim.RMSprop(train_models['alstm'][0].parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, momentum=args.momentum))
    train_models['alstm'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

if 'lmn' in args.models:
    train_models['lmn'].append(LMN(input_size, args.hidden_sizes_lmn, output_size, args.memory_size_lmn,
        device, args.batch_size, type_A=args.type_A))
    train_models['lmn'].append(torch.optim.RMSprop(train_models['lmn'][0].parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, momentum=args.momentum))
    train_models['lmn'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

if 'almn' in args.models:
    train_models['almn'].append(AugmentedLMN(input_size, args.hidden_sizes_lmn, output_size,
        args.memory_size_lmn, device, args.batch_size, type_A=args.type_A, feed_mem=args.feed_mem))
    train_models['almn'].append(torch.optim.RMSprop(train_models['almn'][0].parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, momentum=args.momentum))
    train_models['almn'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

if args.load:
    if 'alstm' in args.models or 'almn' in args.models:
        train_autoencoders[0] = load_autoencoders(train_autoencoders[0], device, os.path.join(plot_folder, path_save_models))
    for model in args.models:
        train_models[model] = load_models(train_models[model], model, os.path.join(plot_folder, path_save_models), device)



def allocate_separate_models(input_size, output_size, args, device):
    separate_models = defaultdict(list)

    separate_models['lmn_sep'].append(LMN(input_size, args.hidden_sizes_lmn, output_size, args.memory_size_lmn,
        device, args.batch_size, type_A=args.type_A))
    separate_models['lmn_sep'].append(torch.optim.RMSprop(separate_models['lmn_sep'][0].parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, momentum=args.momentum))
    separate_models['lmn_sep'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

    separate_models['lstm_sep'].append(BaselineRNN(input_size, args.hidden_size_rnn, output_size, device,
        batch_size=args.batch_size, lstm=True, num_layers=args.layers_rnn, bidirectional=args.bidirectional))
    separate_models['lstm_sep'].append(torch.optim.RMSprop(separate_models['lstm_sep'][0].parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, momentum=args.momentum))
    separate_models['lstm_sep'].append(torch.nn.CrossEntropyLoss(reduction='mean'))

    return separate_models

reconstruction_loss = torch.nn.MSELoss()

################################### Train loop

### Load data
x_all, y_all, _ = load_all_data('tasks/audioset/data/packed_features/bal_train.h5')

if args.unbalanced:
    x_unbal, y_unbal = load_unbal_data('tasks/audioset/data/packed_features/unbal_train.h5', tasks, 10000)

if (not args.not_test) or (not args.not_intermediate_test):
    x_all_test, y_all_test, _ = load_all_data('tasks/audioset/data/packed_features/eval.h5')

intermediate_test_acc = defaultdict(lambda: defaultdict(list))
intermediate_test_loss = defaultdict(lambda: defaultdict(list))


if not args.load:
    for cat_id_p, cat in enumerate(tasks):


        losses = defaultdict(list)
        accs = defaultdict(list)

        accs_val = defaultdict(list)
        ls_val = defaultdict(list)

        avg_loss = defaultdict(float)
        avg_acc = defaultdict(float)

        task_re.append([])

        idx_task = select_category(cat, y_all, one_category=True)
        if idx_task is None:
            print("No data respect the constraint!!")

        x_all_task, y_all_task = x_all[idx_task], y_all[idx_task]

        y_all_task = transform_labels(y_all_task)

        if args.unbalanced:
            y_unbal_cat = transform_labels(y_unbal[cat_id_p], formatted=True)
            x_all_task = torch.cat((x_all_task, x_unbal[cat_id_p]))
            y_all_task = torch.cat( ( y_all_task, y_unbal_cat ) )

        x_train, x_val, y_train, y_val = split_audioset(x_all_task, y_all_task, test_size=0.2)

        print("Examples for training: ", x_train.size(0))
        print("Examples for validation: ", x_val.size(0))

        if small_version:
            epochs = args.epochs[0]
        else:
            epochs = args.epochs[0] if cat_id_p == 0 else args.epochs[1]


        if args.separate_modules:
            separate_models = allocate_separate_models(input_size, output_size, args, device)


        for epoch in range(1, epochs+1):

            if ((epoch-1) % args.print_every == 0) or (epoch == epochs):
                print("Task ", str(cat_id_p), " - Epoch ", epoch, "/", epochs)


                for model in args.models:
                    a_val, l_val = test(train_models, model, x_val,y_val, accuracy, device, output_size)

                    losses[model].append(avg_loss[model] / float(args.print_every))
                    accs[model].append(avg_acc[model] / float(args.print_every))

                    accs_val[model].append(a_val)
                    ls_val[model].append(l_val)

                    print(model, "- Training acc: ", accs[model][-1])
                    print(model, " - Validation acc: ", a_val)
                    print(model, "- Training loss: ", losses[model][-1])
                    print(model, " - Validation loss: ", l_val)

                if args.separate_modules:
                    for model in separate_models.keys():
                        a_val, l_val = test(separate_models, model, x_val,y_val, accuracy, device, output_size)

                        losses[model].append(avg_loss[model] / float(args.print_every))
                        accs[model].append(avg_acc[model] / float(args.print_every))

                        accs_val[model].append(a_val)
                        ls_val[model].append(l_val)

                avg_loss = defaultdict(float)
                avg_acc = defaultdict(float)

            ####################### Training

            x,y = random_segments(x_train, y_train, args.batch_size)

            if cat_id_p > 0 or small_version:
                if len(train_autoencoders) > 0:
                    re = train_autoencoder(train_autoencoders[0][cat_id_p], train_autoencoders[1][cat_id_p], x, reconstruction_loss, device)
                    task_re[-1].append(re)


            for model in args.models:
                
                if (model == 'almn' or model == 'alstm') or (args.ewc_lambda == 0):
                    l, a = train(train_models, model, x,y, accuracy, device, output_size, args.max_grad_norm)
                else:
                    l, a = train_ewc(ewc, cat_id_p, train_models, model, x,y, accuracy, device, output_size, args.max_grad_norm)

                avg_loss[model] += l
                avg_acc[model] += a

            if args.separate_modules:
                for model in separate_models.keys():
                    l, a = train(separate_models, model, x,y, accuracy, device, output_size, args.max_grad_norm)

                    avg_loss[model] += l
                    avg_acc[model] += a


        ####################### End of current task

        for model in args.models:
            task_losses[model].append(losses[model])
            task_acc[model].append(accs[model])

            task_acc_val[model].append(accs_val[model])
            task_l_val[model].append(ls_val[model])

        if args.separate_modules:
            for model in separate_models.keys():
                task_losses[model].append(losses[model])
                task_acc[model].append(accs[model])

                task_acc_val[model].append(accs_val[model])
                task_l_val[model].append(ls_val[model])


        if not args.not_intermediate_test:
            tasks_to_do = tasks[:cat_id_p]
            
            for cat_id_it, cat in enumerate(tasks_to_do):
                idx_task = select_category(cat, y_all_test, one_category=True)
                if idx_task is None:
                    print("No data respect the constraint!!")

                x_all_test_task, y_all_test_task = x_all_test[idx_task], y_all_test[idx_task]
                y_all_test_task = transform_labels(y_all_test_task)

                if len(train_autoencoders) > 0:
                    reconstruction_errors, module_id = test_autoencoder(train_autoencoders,x_all_test_task, reconstruction_loss, device, small_version)
                else:
                    module_id = None

                for model in args.models:
                    a_test, l_test = test(train_models, model, x_all_test_task,y_all_test_task, accuracy, device, output_size, module_id=module_id)
                
                    intermediate_test_acc[model][cat_id_it+1].append(a_test)
                    intermediate_test_loss[model][cat_id_it+1].append(l_test)



        if args.ewc_lambda > 0:
            for model in args.models:
                if model != 'almn' and model != 'alstm':
                    x,y = random_segments(x_train, y_train, 1000)
                    x, y = x.to(device), y.to(device)

                    accumulate_backward(train_models, model, x, y, device, output_size, args.max_grad_norm)

                    ewc.compute_fisher(train_models[model][0], model, cat_id_p)


        if 'almn' in args.models:
            if cat != tasks[-1]: # not at the end
                if task_acc_val['almn'][-1][-1] < args.threshold_acc:
                    print('Adding module LMN', len(train_models['almn'][0].lmns)+1)
                    train_models['almn'][0].add_new_module(train_models['almn'][1]) # same configuration as previous module

        if 'alstm' in args.models:
            if cat != tasks[-1]: # not at the end
                if task_acc_val['alstm'][-1][-1] < args.threshold_acc:
                    print('Adding module LSTM', len(train_models['alstm'][0].lstms)+1)
                    train_models['alstm'][0].add_new_module(train_models['alstm'][1]) # same configuration as previous module


    if args.save:
        for model in args.models:
            save_model(train_models[model][0], model, os.path.join(plot_folder, path_save_models))

    # save autoencoders
    if ('alstm' in args.models or 'almn' in args.models) and args.save:
        save_autoencoders(train_autoencoders[0], os.path.join(plot_folder, path_save_models))


if not args.not_test:
    module_selection_accuracy = np.zeros( (len(tasks), len(tasks)) )

    test_accuracies = defaultdict(list)
    test_losses = defaultdict(list)

    tasks_to_do = tasks if small_version else tasks[1:]
    reconstruction_errors_list = []

    for t_index, cat in enumerate(tasks_to_do):
        idx_task = select_category(cat, y_all_test, one_category=True)
        if idx_task is None:
            print("No data respect the constraint!!")

        x_all_test_task, y_all_test_task = x_all_test[idx_task], y_all_test[idx_task]

        y_all_test_task = transform_labels(y_all_test_task)

        print("Examples for test: ", x_all_test_task.size(0))

        if len(train_autoencoders) > 0:
            reconstruction_errors, module_id = test_autoencoder(train_autoencoders,x_all_test_task, reconstruction_loss, device, small_version)
            #print(reconstruction_errors)
            #print("Choosing module ", module_id)
            reconstruction_errors_list.append(reconstruction_errors)
            module_selection_accuracy[t_index, module_id] += 1
        else:
            module_id = None
            reconstruction_errors_list = []

        for model in args.models:

            a_test, l_test = test(train_models, model, x_all_test_task,y_all_test_task, accuracy, device, output_size, module_id=module_id)

            test_accuracies[model].append(a_test)
            test_losses[model].append(l_test)

    print("Accuracies: ", test_accuracies)
    print("Losses: ", test_losses)

    if not args.not_intermediate_test:
        write_intermediate_test_results(plot_folder, intermediate_test_acc, intermediate_test_loss)

    module_selection_accuracy = module_selection_accuracy / np.sum(module_selection_accuracy, axis=1).reshape(-1,1)
    write_test_results(plot_folder, test_accuracies, test_losses, reconstruction_errors_list, module_selection_accuracy)


################################### Plot and save results

if not args.load:
    if small_version:
        info_plot_small = [list(range(len(tasks))) ,plot_folder, args.epochs[0]]
        info_plot = info_plot_small
    else:
        info_plot_large0 = [list(range(len(tasks))) ,plot_folder, args.epochs[0]]
        info_plot_large1 = [list(range(len(tasks))) ,plot_folder, args.epochs[1]]
        info_plot = info_plot_large0

    if not small_version:
        for model in args.models:
            plot(info_plot_large0, [task_acc[model][0]], model, 'acc0', [task_acc_val[model][0]])
            plot(info_plot_large0, [task_losses[model][0]], model, 'loss0', [task_l_val[model][0]])
            plot(info_plot_large1, task_acc[model][1:], model, 'acc', task_acc_val[model][1:])
            plot(info_plot_large1, task_losses[model][1:], model, 'loss', task_l_val[model][1:])
        if args.separate_modules:
            for model in separate_models.keys():
                plot(info_plot_large0, [task_acc[model][0]], model, 'acc0', [task_acc_val[model][0]])
                plot(info_plot_large0, [task_losses[model][0]], model, 'loss0', [task_l_val[model][0]])
                plot(info_plot_large1, task_acc[model][1:], model, 'acc', task_acc_val[model][1:])
                plot(info_plot_large1, task_losses[model][1:], model, 'loss', task_l_val[model][1:])
    else:
        for model in args.models:
            plot(info_plot_small, task_acc[model], model, 'acc', task_acc_val[model])
            plot(info_plot_small, task_losses[model], model, 'loss', task_l_val[model])
        if args.separate_modules:
            for model in separate_models.keys():
                plot(info_plot_small, task_acc[model], model, 'acc', task_acc_val[model])
                plot(info_plot_small, task_losses[model], model, 'loss', task_l_val[model])


    if len(train_autoencoders) > 0:
        plot(info_plot, task_re, 'autoencoders', 'Reconstruction error')

    with open(plot_folder+'acc','wb') as f:
        pickle.dump(task_acc, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(plot_folder+'loss','wb') as f:
        pickle.dump(task_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(plot_folder+'acc_val','wb') as f:
        pickle.dump(task_acc_val, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(plot_folder+'loss_val','wb') as f:
        pickle.dump(task_l_val, f, protocol=pickle.HIGHEST_PROTOCOL)
    # task_losses = pickle.load(f)

    write_configuration(args, plot_folder)


    if args.cuda:
        print("Plots saved")
    else:
        print("Plots showed")
