import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

class MNIST():
    def __init__(self):
        
        # load the data (training set)
        train_set = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

        x = []
        c = []
        for images, classes in train_loader:
            x += [images]
            c += [classes]

        # just concatenate the test set since we are not interested in classification
        test_set = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transforms.ToTensor()) 
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

        for images, classes in test_loader:
            x += [images]
            c += [classes]

        x = torch.cat(x)
        # make it binary
        x = (x>0.4375).float()
        c = torch.cat(c)

        # split images according to digits
        self.x = []
        for i in range(10):
            self.x += [x[c == i]]

        self.period = 500
        self.count = 0

    def get_batch(self,
                  ind : int, # digit
                  bs : int, # how many samples
                  device : int) -> torch.Tensor:

        # shufftle it once awhile (each digit individually)
        if (self.count % self.period) == 0:
            for i in range(10):
                n = self.x[i].shape[0]
                permarray = torch.randperm(n)
                self.x[i] = self.x[i][permarray]

        # get a portions of data
        n = self.x[ind].shape[0]
        pos = np.random.randint(n-bs)
        x = self.x[ind][pos:pos+bs].to(device)
        self.count += 1

        return x
