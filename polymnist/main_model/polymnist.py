import torch
import numpy as np
import pickle

# Note, the files polymnist-train.pck and polymnist-test.pck should be prepared in advance
# see README-s

class POLYMNIST():
    def __init__(self,
                 bs : int):

        self.bs = bs

        with open('../data/polymnist-train.pck', 'rb') as f:
            data = pickle.load(f)

        x = torch.from_numpy(data['x'])
        # normalize
        self.means = x.mean([0,2,3], keepdim=True)
        x = x - self.means
        self.std = torch.sqrt((x ** 2).mean([0,2,3], keepdim=True))
        self.x_train = x / self.std

        c = torch.from_numpy(data['c']).long()
        self.c_train = torch.nn.functional.one_hot(c, num_classes=10).float()

        d = torch.from_numpy(data['d']).long()
        self.d_train = torch.nn.functional.one_hot(d, num_classes=5).float()

        self.n_train = self.x_train.shape[0]
        self.period_train = self.n_train // bs
        self.count_train = 0

        #################################

        with open('../data/polymnist-test.pck', 'rb') as f:
            data = pickle.load(f)

        x = torch.from_numpy(data['x'])
        # normalize
        x = x - self.means
        self.x_val = x / self.std

        c = torch.from_numpy(data['c']).long()
        self.c_val = torch.nn.functional.one_hot(c, num_classes=10).float()

        d = torch.from_numpy(data['d']).long()
        self.d_val = torch.nn.functional.one_hot(d, num_classes=5).float()

        self.n_val = self.x_val.shape[0]
        self.period_val = self.n_val // bs
        self.count_val = 0

    def get_train(self,
                  device : int) -> torch.Tensor:
        
        if (self.count_train % self.period_train) == 0:
            permarray = torch.randperm(self.n_train)
            self.c_train = self.c_train[permarray]
            self.d_train = self.d_train[permarray]
            self.x_train = self.x_train[permarray]

        # get portions of data
        pos = np.random.randint(self.n_train-self.bs)
        c = self.c_train[pos:pos+self.bs].to(device)
        d = self.d_train[pos:pos+self.bs].to(device)
        x = self.x_train[pos:pos+self.bs].to(device)
        
        self.count_train += 1

        return c, d, x

    def get_val(self,
                device : int) -> torch.Tensor:
        
        if (self.count_val % self.period_val) == 0:
            permarray = torch.randperm(self.n_val)
            self.c_val = self.c_val[permarray]
            self.d_val = self.d_val[permarray]
            self.x_val = self.x_val[permarray]

        # get portions of data
        pos = np.random.randint(self.n_val-self.bs)
        c = self.c_val[pos:pos+self.bs].to(device)
        d = self.d_val[pos:pos+self.bs].to(device)
        x = self.x_val[pos:pos+self.bs].to(device)
        
        self.count_val += 1

        return c, d, x
