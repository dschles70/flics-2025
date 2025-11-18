import torch
from nets import EncoderS

# c -- digit
# d -- style
# z -- latent
# s -- segmentation (binarized MNIST)
# x -- images (PolyMNIST)

# below is implementation for p(z|c, d, s, x)

class ZNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 nz : int):

        super(ZNet, self).__init__()

        self.net = EncoderS(4, nc + nd, nz)

    def forward(self,
                c : torch.Tensor,
                d : torch.Tensor,
                s : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:
         
         cd = torch.cat((c, d), dim=1)
         sx = torch.cat((s, x), dim=1)
         return self.net(sx, cd)

class ZModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 nz : int,
                 step : float):

        super(ZModel, self).__init__()

        self.nz = nz

        self.net = ZNet(nc, nd, nz)

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

        # filter out those examples, where z was just sampled (expected gradient should be zero)        
        c = c[l!=0]
        d = d[l!=0]
        z = z[l!=0]
        s = s[l!=0]
        x = x[l!=0]
        
        self.optimizer.zero_grad()

        scores = self.net(c, d, s, x)
        loss = self.criterion(scores, z).sum(1).mean()
        loss.backward()

        self.optimizer.step()

        return loss.detach() / self.nz

    def sample(self,
               c : torch.Tensor,
               d : torch.Tensor,
               s : torch.Tensor,
               x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(c, d, s, x)
            return torch.sigmoid(scores).bernoulli().detach()

    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        return (torch.rand([bs, self.nz], device=self.dummy_param.device) < 0.5).float()
