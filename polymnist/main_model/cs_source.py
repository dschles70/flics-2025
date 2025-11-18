import torch
from mnist import MNIST

# this is a wrapper for either real or synthetic MNIST-like samples
class CSSOURCE():
    def __init__(self,
                 bs : int,
                 generator_path : str, # if empty, real MNIST
                 device : int):
        
        self.bs = bs

        if generator_path != '':
            self.mode = 1
            self.model = torch.jit.load(generator_path).to(device)
            print('CSSOURCE: using synthetic data', flush=True)
        else:
            self.mode = 0
            self.mnist = MNIST(bs)
            print('CSSOURCE: using real data', flush=True)
    
    def get_train(self,
                  device : int):
        
        if self.mode:
            c = torch.randint(10, [self.bs])
            c = torch.nn.functional.one_hot(c, num_classes=10).float().to(device)
            x = self.model.limiting(c)
            return c, x
        else:
            return self.mnist.get_train(device)

    def get_val(self,
                device : int):
        
        if self.mode:
            c = torch.randint(10, [self.bs])
            c = torch.nn.functional.one_hot(c, num_classes=10).float().to(device)
            x = self.model.limiting(c)
            return c, x
        else:
            return self.mnist.get_val(device)
