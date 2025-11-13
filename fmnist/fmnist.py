import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

class FMNIST():
    def __init__(self,
                 bs : int):
        
        self.bs = bs

        # training set
        train_set = torchvision.datasets.FashionMNIST("./data", download=True, train=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

        x = []
        c = []
        for images, classes in train_loader:
            x += [images]
            c += [classes]

        self.x_train = torch.cat(x)
        self.c_train = torch.cat(c)
        self.n_train = self.x_train.shape[0]
        self.period_train = self.n_train//100
        self.count_train = 0

        # validation set
        test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.ToTensor()) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

        x = []
        c = []
        for images, classes in test_loader:
            x += [images]
            c += [classes]

        self.x_val = torch.cat(x)
        self.c_val = torch.cat(c) # original labels
        self.n_val = self.x_val.shape[0]
        self.period_val = self.n_val//100
        self.count_val = 0

        # training set split into digits, heavily reduced
        self.x_train_w = []
        for i in range(10):
            x_i = self.x_train[self.c_train == i]
            perm = torch.randperm(x_i.shape[0])
            x_i = x_i[perm][0:bs*4] # hard coded, we have 4 models
            self.x_train_w += [x_i]

    def get_train(self,
                  bs : int,
                  device : int) -> tuple:
        
        if (self.count_train % self.period_train) == (self.period_train - 1):
            permarray = torch.randperm(self.n_train)
            self.x_train = self.x_train[permarray]
            self.c_train = self.c_train[permarray]

        # get portions of data
        pos = np.random.randint(self.n_train-bs)
        x = self.x_train[pos:pos+bs].to(device)
        c = self.c_train[pos:pos+bs].to(device)
        c = torch.nn.functional.one_hot(c, num_classes=10).float()

        self.count_train += 1

        return c, x

    def get_val(self,
                bs : int,
                device : int) -> tuple:
        
        if (self.count_val % self.period_val) == (self.period_val - 1):
            permarray = torch.randperm(self.n_val)
            self.x_val = self.x_val[permarray]
            self.c_val = self.c_val[permarray]

        # get portions of data
        pos = np.random.randint(self.n_val-bs)
        x = self.x_val[pos:pos+bs].to(device)
        c = self.c_val[pos:pos+bs].to(device)
        c = torch.nn.functional.one_hot(c, num_classes=10).float()

        self.count_val += 1

        return c, x

    def get_train_w(self,
                    ind : int,
                    device : int) -> tuple:

        c = [torch.ones([self.bs], device=device) * i for i in range(10)]
        c = torch.cat(c).long()
        c = torch.nn.functional.one_hot(c, num_classes=10).float()

        x = []
        for i in range(10):
            x += [self.x_train_w[i][ind*self.bs:ind*self.bs+self.bs]]
        x = torch.cat(x).to(device)

        return c, x
