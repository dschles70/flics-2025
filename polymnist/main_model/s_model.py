import torch
from nets import UNetS

# c -- digit
# d -- style
# z -- latent
# s -- segmentation (binarized MNIST)
# x -- images (PolyMNIST)

# below is implementation for p(s | c, d, z, x)

class SNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 nz : int):

        super(SNet, self).__init__()

        self.net = UNetS(3, nc + nd + nz, 1)

    def forward(self,
                c : torch.Tensor,
                d : torch.Tensor,
                z : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:
         
         cdz = torch.cat((c, d, z), dim=1)
         return self.net(x, cdz)

class SModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 nz : int,
                 step : float):

        super(SModel, self).__init__()

        self.net = SNet(nc, nd, nz)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.dummy_param = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    def optimize(self,
                 c : torch.Tensor,
                 d : torch.Tensor,
                 z : torch.Tensor,
                 s : torch.Tensor,
                 x : torch.Tensor,
                 l : torch.Tensor) -> tuple:
        
        # filter out those examples, where s was just sampled (expected gradient should be zero)        
        c = c[l!=1]
        d = d[l!=1]
        z = z[l!=1]
        s = s[l!=1]
        x = x[l!=1]

        self.optimizer.zero_grad()

        scores = self.net(c, d, z, x)
        loss = self.criterion(scores, s).sum([1,2,3]).mean()
        loss.backward()

        self.optimizer.step()

        return loss.detach() / (28 * 28)

    def sample(self,
               c : torch.Tensor,
               d : torch.Tensor,
               z : torch.Tensor,
               x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(c, d, z, x)
            return torch.sigmoid(scores).bernoulli().detach()

    def get_dec(self,
                c : torch.Tensor,
                d : torch.Tensor,
                z : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(c, d, z, x)
            return (scores > 0).float().detach()

    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        return (torch.rand([bs, 1, 28, 28], device=self.dummy_param.device) < 0.5).float()
