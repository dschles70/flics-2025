import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

class MNIST():
    def __init__(self,
                 bs : int):

        self.bs = bs

        # load data (training set)
        train_set = torchvision.datasets.MNIST("../data", download=True, train=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

        x = []
        c = []
        for images, classes in train_loader:
            x += [images]
            c += [classes]
        x = torch.cat(x)
        self.x_train = (x>0.4375).float()
        c = torch.cat(c)
        self.c_train = torch.nn.functional.one_hot(c, num_classes=10).float()

        self.n_train = self.x_train.shape[0]
        self.period_train = self.n_train // bs
        self.count_train = 0

        # load data (validation set)
        val_set = torchvision.datasets.MNIST("../data", download=True, train=False, transform=transforms.ToTensor()) 
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=100)

        x = []
        c = []
        for images, classes in val_loader:
            x += [images]
            c += [classes]
        x = torch.cat(x)
        self.x_val = (x>0.4375).float()
        c = torch.cat(c)
        self.c_val = torch.nn.functional.one_hot(c, num_classes=10).float()

        self.n_val = self.x_val.shape[0]
        self.period_val = self.n_val // bs
        self.count_val = 0

    def get_train(self,
                  device : int) -> torch.Tensor:
        
        if (self.count_train % self.period_train) == 0:
            permarray = torch.randperm(self.n_train)
            self.x_train = self.x_train[permarray]
            self.c_train = self.c_train[permarray]

        # get portions of data
        pos = np.random.randint(self.n_train-self.bs)
        x = self.x_train[pos:pos+self.bs].to(device)
        c = self.c_train[pos:pos+self.bs].to(device)
        
        self.count_train += 1

        return c, x

    def get_val(self,
                device : int) -> torch.Tensor:
        
        if (self.count_val % self.period_val) == 0:
            permarray = torch.randperm(self.n_val)
            self.x_val = self.x_val[permarray]
            self.c_val = self.c_val[permarray]

        # get portions of data
        pos = np.random.randint(self.n_val-self.bs)
        x = self.x_val[pos:pos+self.bs].to(device)
        c = self.c_val[pos:pos+self.bs].to(device)
        
        self.count_val += 1

        return c, x
