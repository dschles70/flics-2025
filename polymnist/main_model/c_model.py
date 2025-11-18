import torch
from nets import EncoderS

# c -- digit
# d -- style
# z -- latent
# s -- segmentation (binarized MNIST)
# x -- images (PolyMNIST)

# below is implementation for p(c|d, z, s, x)

class CNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 nz : int):

        super(CNet, self).__init__()

        self.net = EncoderS(4, nd + nz, nc)

    def forward(self,
                d : torch.Tensor,
                z : torch.Tensor,
                s : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:
         
         dz = torch.cat((d, z), dim=1)
         sx = torch.cat((s, x), dim=1)
         return self.net(sx, dz)

class CModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 nz : int,
                 step : float):

        super(CModel, self).__init__()

        self.nc = nc

        self.net = CNet(nc, nd, nz)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.dummy_param = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    def optimize(self,
                 c : torch.Tensor,
                 d : torch.Tensor,
                 z : torch.Tensor,
                 s : torch.Tensor,
                 x : torch.Tensor) -> tuple:
        
        self.optimizer.zero_grad()

        scores = self.net(d, z, s, x)
        loss = self.criterion(scores, c).mean()
        loss.backward()

        self.optimizer.step()

        return loss.detach()

    def sample(self,
               d : torch.Tensor,
               z : torch.Tensor,
               s : torch.Tensor,
               x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(d, z, s, x)
            distr = torch.distributions.one_hot_categorical.OneHotCategorical(logits=scores)
            return distr.sample()

    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        ind = torch.randint(self.nc, [bs], device=self.dummy_param.device)
        return torch.nn.functional.one_hot(ind, num_classes=self.nc).float()
