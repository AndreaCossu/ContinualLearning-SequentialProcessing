'''
This script executes the experiments on the MNIST dataset by
training the chosen models to learn to predict a single digit after having seen
its complete sequence in input.
'''

import torch
import argparse
import pickle
import os
import numpy as np

from tasks.dataset_cl import MNIST_CL, FashionMNIST_CL, CIFAR10_CL, Devanagari_CL
from torch.utils.data import DataLoader
from core.EWC_wrapper import EWC
from tasks.mnist.utils_single import *
from tasks.utils import *
from collections import defaultdict
from experiment.CL_experiment import CL_Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2, help='epochs to train.')
parser.add_argument('--hidden_size_rnn', type=int, default=128, help='units of RNN')
parser.add_argument('--layers_rnn', type=int, default=1, help='layers of RNN')
parser.add_argument('--models', nargs='+', type=str, default=['lstm'], help='models to train: lstm, alstm, rnn, lmn, almn')
parser.add_argument('--tasks', nargs='+', type=int, default=[1,2,3,4,5], help='Task to train: from 1 to 5. Pair of digit')
parser.add_argument('--output_size', type=int, default=2, help='model output size')
parser.add_argument('--input_size', type=int, default=1, help='model input size')


parser.add_argument('--hidden_sizes_lmn', nargs='+', type=int, default=[128], help='layers of functional component of LMN')
parser.add_argument('--hidden_size_autoencoder', type=int, default=500, help='hidden size of the autoencoders')
parser.add_argument('--lr_ae', type=float, default=1e-4, help='optimizer hyperparameter')
parser.add_argument('--decay_ae', type=float, default=0., help='optimizer hyperparameter')

parser.add_argument('--memory_size_lmn', type=int, default=128, help='memory size of LMN')
parser.add_argument('--type_A', action="store_true", help='choose LMN-A ')
parser.add_argument('--feed_mem', action="store_true", help='feed previous memory module to current memory module')
parser.add_argument('--threshold_acc', type=float, default=1.01, help='add new ALMN module if test accuracy is below this threshold')

parser.add_argument('--orthogonal', action="store_true", help='Use orthogonal matrixes for LMN and ALMN')
parser.add_argument('--orthogonal_loss', action="store_true", help='Use orthogonal loss to keep matrixes orthogonal for LMN and ALMN')
parser.add_argument('--lamb', type=float, default=0.1, help='orthogonal regularization hyperparameter')

parser.add_argument('--monitor', action="store_true", help='Monitor orthogonality')

parser.add_argument('--permutation', type=int, default=0, help='Choose permutation of pixel. From 1 to 10. 0 to disable.')
parser.add_argument('--ewc_lambda', type=float, default=0.0, help='EWC task specific hyper parameter. 0 to disable EWC.')

parser.add_argument('--devanagari', action="store_true", help='Use Devanagari dataset.')
parser.add_argument('--fashion', action="store_true", help='Use FashionMNIST dataset.')
parser.add_argument('--cifar', action="store_true", help='Use CIFAR10 dataset.')

# optimizer parameters
parser.add_argument('--weight_decay', type=float, default=0, help='optimizer hyperparameter')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='optimizer hyperparameter')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Value to clip gradient norm.')

parser.add_argument('--tenclasses', action="store_true", help='Use full MNIST with ten classes.')

parser.add_argument('--not_test', action="store_true", help='disable final test')
parser.add_argument('--not_intermediate_test', action="store_true", help='disable final test')

parser.add_argument('--save', action="store_true", help='save models')
parser.add_argument('--load', action="store_true", help='load models')


parser.add_argument('--cuda', action="store_true", help='use gpu')
parser.add_argument('--plot_folder', type=str, default='plots/test/', help='folder in which to put saved plots. Created if not existing.')

args = parser.parse_args()
################################### Init params and folders

cl_exp = CL_Experiment(args)

device = cl_exp.get_device()

plot_folder = cl_exp.configure_plots()

if args.monitor:
    writer = cl_exp.get_writer()

if args.orthogonal_loss:
    args.orthogonal = True

if args.tenclasses:
    args.tasks = [1]
    args.output_size = 10


# perm = torch.randperm(28*28)
# select 1 permutation from the 10 possible permutations
'''
to save permutations:
l = torch.stack([ torch.randperm(28*28) for i in range(10) ]).numpy() 
with open('file.npy', 'wb') as f:
    np.save(f, l)
'''

perm = torch.from_numpy(np.load('tasks/mnist/permutations.npy'))[args.permutation-1] 
perm_cifar = torch.from_numpy(np.load('tasks/mnist/permutations_cifar.npy'))[args.permutation-1]


################################### Create models

if args.ewc_lambda > 0:
    ewc = EWC(device, lamb=args.ewc_lambda)

if args.cifar:
    args.input_size = 3 # RGB channels


train_models, train_autoencoders = cl_exp.create_models()

reconstruction_loss = MSEMasked

################################### Train loop

task_losses = defaultdict(list)
task_acc = defaultdict(list)
task_accs_val = defaultdict(list)
task_losses_val = defaultdict(list)

intermediate_test_acc = defaultdict(lambda: defaultdict(list))
intermediate_test_loss = defaultdict(lambda: defaultdict(list))

task_re = []

# 5 subtasks
if args.tenclasses:
    tt = [list(range(10))]
else:
    tt = [[0,1],[2,3],[4,5],[6,7],[8,9]]



dataroot = 'tasks/mnist/data'
if args.fashion:
    mnist_dataset_train = FashionMNIST_CL(dataroot, download=True, train=True, perc_val=0.25, batch_size=args.batch_size, output_size=args.output_size)
    mnist_dataset_test = FashionMNIST_CL(dataroot, download=True, train=False, output_size=args.output_size)
elif args.cifar:
    mnist_dataset_train = CIFAR10_CL(dataroot, download=True, train=True, perc_val=0.25, batch_size=args.batch_size, output_size=args.output_size)
    mnist_dataset_test = CIFAR10_CL(dataroot, download=True, train=False, output_size=args.output_size)
elif args.devanagari:
    dev_cl = Devanagari_CL(args.batch_size, 'tasks/mnist/data/Devanagari_CL')
    mnist_dataset_train_all = dev_cl.get_subtasks(True, args.tasks)
    mnist_dataset_test_all = dev_cl.get_subtasks(False, args.tasks)
else:
    mnist_dataset_train = MNIST_CL(dataroot, download=True, train=True, perc_val=0.25, batch_size=args.batch_size, output_size=args.output_size)
    mnist_dataset_test = MNIST_CL(dataroot, download=True, train=False, output_size=args.output_size)



if not args.load:
    for task_id, task in enumerate(args.tasks):
        
        if args.devanagari:
            mnist_dataset_train = mnist_dataset_train_all[task_id]
            loader_task_train, loader_task_val = dev_cl.get_train_val_loader(mnist_dataset_train)
        else:
            mnist_dataset_train.choose_subset(tt[task-1])
            loader_task_train, loader_task_val = mnist_dataset_train.get_train_val_loader()

        len_train = len(loader_task_train)

        losses = defaultdict(list)
        accs = defaultdict(list)

        accs_val = defaultdict(list)
        ls_val = defaultdict(list)

        avg_loss = defaultdict(float)
        avg_acc = defaultdict(float)

        task_re.append([])

        for epoch in range(1, args.epochs+1):
            print("Task ", task, " - Epoch ", epoch, "/", args.epochs)

            ####################### Print intermediate results
            if args.orthogonal and args.monitor:
                monitor_orthogonality(args, writer, task, epoch, train_models)


            assert(len(loader_task_val) == 1)
            for x,y in loader_task_val: # fake loop, it is only one large batch
                x = x.view(x.size(0), -1, args.input_size)
                if args.permutation > 0:
                    if args.cifar:
                        x = permute_cifar(x, perm_cifar)
                    else:
                        x = permute(x, perm)
                x,y = x.to(device), y.to(device)

                for model in args.models:
                    l_val, a_val = test(train_models, model, x,y, accuracy, device, args.output_size)
                    accs_val[model].append(a_val)
                    ls_val[model].append(l_val)

            ####################### Train

            for id_batch, (x,y) in enumerate(loader_task_train):
                
                x = x.view(x.size(0), -1, args.input_size)
                if args.permutation > 0:
                    if args.cifar:
                        x = permute_cifar(x, perm_cifar)
                    else:
                        x = permute(x, perm)
                x, y = x.to(device), y.to(device)

                # if autoencoders are needed (i.e. if there are augmented models)
                if len(train_autoencoders) > 0:
                    t_id = args.tasks.index(task)
                    re = train_autoencoder(train_autoencoders[0][t_id], train_autoencoders[1][t_id], x, reconstruction_loss, device)
                    task_re[-1].append(re)

                for model in args.models:
                    if (model == 'almn' or model == 'alstm') or (args.ewc_lambda == 0):
                        l, acc = train(train_models, model, x,y, accuracy, device, args.output_size, args.max_grad_norm)
                    else:
                        l, acc = train_ewc(ewc, task_id, train_models, model, x,y, accuracy, device, args.output_size, args.max_grad_norm)

                    avg_loss[model] += l
                    avg_acc[model] += acc

            if ('alstm' in args.models) and task_id > 0 and args.monitor:
                visualize1 = train_models['alstm'][0].lstms[-1].lstm.all_weights[0][0][:, (args.input_size+1):].cpu()
                visualize2 = train_models['alstm'][0].lstms[-1].lstm.all_weights[0][0][:, :args.input_size].cpu()

                writer.add_image("alstm_connections_additional/"+str(task_id), visualize1, epoch, dataformats='HW')
                writer.add_image("alstm_connections_input/"+str(task_id), visualize2, epoch, dataformats='HW')

            # print metrics
            for model in args.models:
                losses[model].append(avg_loss[model] / float(len_train))
                accs[model].append(avg_acc[model] / float(len_train))

                print(model, "- Training accuracy: ", accs[model][-1])
                print(model, " - Validation accuracy: ", accs_val[model][-1])
                print(model, "- Training loss: ", losses[model][-1])
                print(model, " - Validation loss: ", ls_val[model][-1])

            avg_loss = defaultdict(float)
            avg_acc = defaultdict(float)



        ####################### End of current task

        assert(len(loader_task_val) == 1)
        for x,y in loader_task_val: # fake loop, it is only one large batch
            print("Final validation on task ", task)

            x = x.view(x.size(0), -1, args.input_size)
            if args.permutation > 0:
                if args.cifar:
                    x = permute_cifar(x, perm_cifar)
                else:
                    x = permute(x, perm)
            x,y = x.to(device), y.to(device)

            for model in args.models:
                l_val, a_val = test(train_models, model, x,y, accuracy, device, args.output_size)
                print(model, " - Validation accuracy: ", a_val)
                print(model, " - Validation loss: ", l_val)
                accs_val[model].append(a_val)
                ls_val[model].append(l_val)


        for model in args.models:
            task_losses[model].append(losses[model])
            task_acc[model].append(accs[model])

            task_accs_val[model].append(accs_val[model])
            task_losses_val[model].append(ls_val[model])


        if not args.not_intermediate_test:
            for int_task in range(task_id+1):

                if args.devanagari:
                    mnist_dataset_test = mnist_dataset_test_all[int_task]
                    loader_task_test = DataLoader(mnist_dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)
                else:
                    mnist_dataset_test.choose_subset(tt[int_task])
                    loader_task_test = DataLoader(mnist_dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

                avg_test_accuracies = defaultdict(float)
                avg_test_losses = defaultdict(float)

                for x,y in loader_task_test: 
                    x = x.view(x.size(0), -1, args.input_size)
                    if args.permutation > 0:
                        if args.cifar:
                            x = permute_cifar(x, perm_cifar)
                        else:
                            x = permute(x, perm)
                    x, y = x.to(device), y.to(device)

                    if len(train_autoencoders) > 0:
                        reconstruction_errors, module_id = test_autoencoder(train_autoencoders, x, reconstruction_loss, device)
                    else:
                        module_id = None
                        reconstruction_errors_list = []

                    for model in args.models:
                        l_test, a_test = test(train_models, model, x,y, accuracy, device, args.output_size, module_id=module_id)

                        avg_test_accuracies[model] += a_test
                        avg_test_losses[model] += l_test

                for model in args.models:
                    intermediate_test_acc[model][int_task+1].append(avg_test_accuracies[model] / float(len(loader_task_test)))
                    intermediate_test_loss[model][int_task+1].append(avg_test_losses[model] / float(len(loader_task_test)))



        if args.ewc_lambda > 0:
            for model in args.models:
                if model != 'almn' and model != 'alstm':
                    x_batch, y_batch = [], []
                    for i, (x,y) in enumerate(loader_task_train):
                        x = x.view(x.size(0), -1, args.input_size)
                        if args.permutation > 0:
                            if args.cifar:
                                x = permute_cifar(x, perm_cifar)
                            else:
                                x = permute(x, perm)
                        x, y = x.to(device), y.to(device)
                        x_batch.append(x)
                        y_batch.append(y)


                    accumulate_backward(train_models, model, x_batch, y_batch, device, args.output_size, args.max_grad_norm)

                    ewc.compute_fisher(train_models[model][0], model, task_id)


        if 'almn' in args.models:
            if task != args.tasks[-1]: # not at the end
                if task_accs_val['almn'][-1][-1] < args.threshold_acc:
                    print('Adding module LMN', len(train_models['almn'][0].lmns)+1)
                    train_models['almn'][0].add_new_module(train_models['almn'][1]) # same configuration as previous module
                    if args.orthogonal_loss:
                        train_models['almn'][2].update_memory(train_models['almn'][0].lmns[-1].memory)

        if 'alstm' in args.models:
            if task != args.tasks[-1]: # not at the end
                if task_accs_val['alstm'][-1][-1] < args.threshold_acc:
                    print('Adding module LSTM', len(train_models['alstm'][0].lstms)+1)
                    train_models['alstm'][0].add_new_module(train_models['alstm'][1]) # same configuration as previous module
                    if args.orthogonal_loss:
                        train_models['alstm'][2].update_memory(train_models['alstm'][0].lstms[-1])

    if args.save:
        for model in args.models:
            save_model(train_models[model][0], model, os.path.join(plot_folder, path_save_models))

    # save autoencoders
    if ('alstm' in args.models or 'almn' in args.models) and args.save:
        save_autoencoders(train_autoencoders[0], os.path.join(plot_folder, path_save_models))




# Test with autoencoders
if not args.not_test:
    module_selection_accuracy = np.zeros( (len(tt), len(tt)) )

    test_accuracies = defaultdict(list)
    test_losses = defaultdict(list)
    reconstruction_errors_list = []

    for task in args.tasks:
        if args.devanagari:
            mnist_dataset_test = mnist_dataset_test_all[task-1]
            loader_task_test = DataLoader(mnist_dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)
        else:
            mnist_dataset_test.choose_subset(tt[task-1])
            loader_task_test = DataLoader(mnist_dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

        avg_test_accuracies = defaultdict(float)
        avg_test_losses = defaultdict(float)

        avg_reconstruction_errors = []

        for x,y in loader_task_test: 
            x = x.view(x.size(0), -1, args.input_size)
            if args.permutation > 0:
                if args.cifar:
                    x = permute_cifar(x, perm_cifar)
                else:
                    x = permute(x, perm)
            x, y = x.to(device), y.to(device)


            if len(train_autoencoders) > 0:
                reconstruction_errors, module_id = test_autoencoder(train_autoencoders, x, reconstruction_loss, device)
                #print(reconstruction_errors)
                #print("Choosing module ", module_id)
                avg_reconstruction_errors.append(reconstruction_errors)
                module_selection_accuracy[task-1, module_id] += 1
            
            else:
                module_id = None
                reconstruction_errors_list = []

            for model in args.models:
                l_test, a_test = test(train_models, model, x,y, accuracy, device, args.output_size, module_id=module_id)

                avg_test_accuracies[model] += a_test
                avg_test_losses[model] += l_test

        if len(train_autoencoders) > 0:
            reconstruction_errors_list.append( np.mean(np.array(avg_reconstruction_errors), axis=0).tolist() )

        for model in args.models:
            test_accuracies[model].append(avg_test_accuracies[model] / float(len(loader_task_test)))
            test_losses[model].append(avg_test_losses[model] / float(len(loader_task_test)))

    if not args.not_intermediate_test:
        write_intermediate_test_results(plot_folder, intermediate_test_acc, intermediate_test_loss)

    module_selection_accuracy = module_selection_accuracy / np.sum(module_selection_accuracy, axis=1).reshape(-1,1)
    write_test_results(plot_folder, test_accuracies, test_losses, reconstruction_errors_list, module_selection_accuracy)

    print("Accuracies: ", test_accuracies)
    print("Losses: ", test_losses)


################################### Plot and save results


if not args.load:
    info_plot = [args.tasks, plot_folder, args.epochs]

    for model in args.models:
        plot(info_plot, task_acc[model], model, 'Accuracy', task_accs_val[model])
        plot(info_plot, task_losses[model], model, 'loss', task_losses_val[model])
    if len(train_autoencoders) > 0:
        plot(info_plot, task_re, 'autoencoders', 'Reconstruction error')


    with open(os.path.join(plot_folder, 'ACC'),'wb') as f:
        pickle.dump(task_acc, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_folder, 'loss'),'wb') as f:
        pickle.dump(task_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_folder, 'ACC_val'),'wb') as f:
        pickle.dump(task_accs_val, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_folder, 'loss_val'),'wb') as f:
        pickle.dump(task_losses_val, f, protocol=pickle.HIGHEST_PROTOCOL)
    # task_losses = pickle.load(f)

    if args.cuda:
        print("Plots saved")
    else:
        print("Plots showed")

write_configuration(args, plot_folder)
if args.monitor:
    writer.close()
