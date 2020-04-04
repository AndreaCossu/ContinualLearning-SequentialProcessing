import torchvision.datasets as datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from sklearn.model_selection import train_test_split
import os

class MNIST_CL(datasets.MNIST):
    def __init__(self, root, download, train, perc_val=0.2, batch_size=3, output_size=None):
        '''
        :param output_size: number of output units of the model
        '''
        
        super(MNIST_CL, self).__init__(root, train=train,download=download, transform=transforms.ToTensor())

        self.all_targets = self.targets
        self.all_data = self.data
        
        self.perc_val = perc_val
        self.batch_size = batch_size

        self.output_size = output_size

        self.train = train
    
    def choose_subset(self, labels):
        '''
        Select a subset of dataset with provided labels.

        :param labels: a list containing integer representing digits to select from dataset
        '''


        mask = torch.stack([ (self.all_targets == l) for l in labels ]) # one mask for each label
        mask = torch.sum(mask, dim=0) # or of masks
        mask = mask.nonzero().squeeze() # index of valid elements

        self.targets = self.all_targets[mask]
        self.data = self.all_data[mask]

        # restrict output targets to output_size values
        if self.output_size is not None:
            self.targets = self.targets % self.output_size
    
    def get_train_val_loader(self):
        '''
        Return an exception if not in training mode.

        :return mnist_cl_loader_train: mini-batch loader for training set
        :return mnist_cl_loader_val: full-batch loader for validation set (one batch with all validation set)
        '''


        if self.train:
            train_indices, val_indices = train_test_split(list(range(self.targets.size(0))), test_size=self.perc_val, shuffle=True, stratify=self.targets.numpy() )
            train_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size=self.batch_size, drop_last=True)
            val_sampler = BatchSampler(SubsetRandomSampler(val_indices), batch_size=len(val_indices), drop_last=False)

            mnist_cl_loader_train = DataLoader(self, batch_sampler=train_sampler)
            mnist_cl_loader_val = DataLoader(self, batch_sampler=val_sampler)
        
            return mnist_cl_loader_train, mnist_cl_loader_val
        else:
            raise Exception("Cannot split train and validation when mode test is on. Split is allowed only in train mode.")
    


class FashionMNIST_CL(datasets.FashionMNIST):
    def __init__(self, root, download, train, perc_val=0.2, batch_size=3, output_size=None):
        '''
        :param output_size: number of output units of the model
        '''
        
        super(FashionMNIST_CL, self).__init__(root, train=train,download=download, transform=transforms.ToTensor())

        self.all_targets = self.targets
        self.all_data = self.data
        
        self.perc_val = perc_val
        self.batch_size = batch_size

        self.output_size = output_size

        self.train = train
    
    def choose_subset(self, labels):
        '''
        Select a subset of dataset with provided labels.

        :param labels: a list containing integer representing digits to select from dataset
        '''


        mask = torch.stack([ (self.all_targets == l) for l in labels ]) # one mask for each label
        mask = torch.sum(mask, dim=0) # or of masks
        mask = mask.nonzero().squeeze() # index of valid elements

        self.targets = self.all_targets[mask]
        self.data = self.all_data[mask]

        # restrict output targets to output_size values
        if self.output_size is not None:
            self.targets = self.targets % self.output_size
    
    def get_train_val_loader(self):
        '''
        Return an exception if not in training mode.

        :return mnist_cl_loader_train: mini-batch loader for training set
        :return mnist_cl_loader_val: full-batch loader for validation set (one batch with all validation set)
        '''


        if self.train:
            train_indices, val_indices = train_test_split(list(range(self.targets.size(0))), test_size=self.perc_val, shuffle=True, stratify=self.targets.numpy() )
            train_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size=self.batch_size, drop_last=True)
            val_sampler = BatchSampler(SubsetRandomSampler(val_indices), batch_size=len(val_indices), drop_last=False)

            mnist_cl_loader_train = DataLoader(self, batch_sampler=train_sampler)
            mnist_cl_loader_val = DataLoader(self, batch_sampler=val_sampler)
        
            return mnist_cl_loader_train, mnist_cl_loader_val
        else:
            raise Exception("Cannot split train and validation when mode test is on. Split is allowed only in train mode.")


class CIFAR10_CL(datasets.CIFAR10):
    def __init__(self, root, download, train, perc_val=0.2, batch_size=3, output_size=None):
        '''
        :param output_size: number of output units of the model
        '''
        
        super(CIFAR10_CL, self).__init__(root, train=train,download=download)

        self.all_targets = torch.tensor(self.targets)
        self.all_data = torch.tensor(self.data).float()
        
        self.perc_val = perc_val
        self.batch_size = batch_size

        self.output_size = output_size

        self.train = train
        
    def choose_subset(self, labels):
        '''
        Select a subset of dataset with provided labels.

        :param labels: a list containing integer representing digits to select from dataset
        '''


        mask = torch.stack([ (self.all_targets == l) for l in labels ]) # one mask for each label
        mask = torch.sum(mask, dim=0) # or of masks
        mask = mask.nonzero().squeeze() # index of valid elements

        self.targets = self.all_targets[mask]
        self.data = self.all_data[mask]

        # restrict output targets to output_size values
        if self.output_size is not None:
            self.targets = self.targets % self.output_size
    
    def get_train_val_loader(self):
        '''
        Return an exception if not in training mode.

        :return mnist_cl_loader_train: mini-batch loader for training set
        :return mnist_cl_loader_val: full-batch loader for validation set (one batch with all validation set)
        '''


        if self.train:
            train_indices, val_indices = train_test_split(list(range(self.targets.size(0))), test_size=self.perc_val, shuffle=True, stratify=self.targets.numpy() )
            train_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size=self.batch_size, drop_last=True)
            val_sampler = BatchSampler(SubsetRandomSampler(val_indices), batch_size=len(val_indices), drop_last=False)

            mnist_cl_loader_train = DataLoader(self, batch_sampler=train_sampler)
            mnist_cl_loader_val = DataLoader(self, batch_sampler=val_sampler)
        
            return mnist_cl_loader_train, mnist_cl_loader_val
        else:
            raise Exception("Cannot split train and validation when mode test is on. Split is allowed only in train mode.")
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = img / 255 # normalize in [0, 1]

        return img, target
    

class Devanagari_CL():

    def __init__(self, batch_size, path='tasks/mnist/data/Devanagari_CL'):
        self.path = path
        self.batch_size = batch_size

    
    def get_subtasks(self, train, tasks):
        if train:
            mode = 'Train'
        else:
            mode = 'Test'

        mnist_dataset = []
        for t in tasks:
            mnist_dataset.append( datasets.ImageFolder(os.path.join(self.path, mode,  str(t)), transform=transforms.Compose([
                transforms.CenterCrop(28),
                transforms.Grayscale(),
                transforms.ToTensor()
            ]))
            )


        return mnist_dataset
    
    def get_train_val_loader(self, dl, perc_val=0.25):
        train_indices, val_indices = train_test_split(list(range(len(dl.targets))), test_size=perc_val, shuffle=True, stratify=dl.targets )
        train_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size=self.batch_size, drop_last=True)
        val_sampler = BatchSampler(SubsetRandomSampler(val_indices), batch_size=len(val_indices), drop_last=False)

        mnist_cl_loader_train = DataLoader(dl, batch_sampler=train_sampler)
        mnist_cl_loader_val = DataLoader(dl, batch_sampler=val_sampler)

        return mnist_cl_loader_train, mnist_cl_loader_val