import torch
from nets import EncoderS, Decoder

# Conditional VAE for p(s|c) with latent binary z,
# s are binarized MNIST images

# p(s|c,z)
class PSNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int):

        super(PSNet, self).__init__()

        self.net = Decoder(nc + nz, 1)

    def forward(self,
                c : torch.Tensor,
                z : torch.Tensor) -> torch.Tensor:
         
         cz = torch.cat((c, z), dim=1)
         return self.net(cz)

# p(z|c)
class PZNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int):

        super(PZNet, self).__init__()

        nn = (nz + nz) * 2
        self.lin1 = torch.nn.Linear(nc, nn)
        self.lin2 = torch.nn.Linear(nn, nn)
        self.lin3 = torch.nn.Linear(nn, nz)

    def forward(self,
                c : torch.Tensor) -> torch.Tensor:
         
         hh = torch.tanh(self.lin1(c))
         hh = torch.tanh(self.lin2(hh))
         return self.lin3(hh)

# q(z|c,s)
class QZNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int):

        super(QZNet, self).__init__()

        self.net = EncoderS(1, nc, nz)

    def forward(self,
                c : torch.Tensor,
                s : torch.Tensor) -> torch.Tensor:
         
         return self.net(s, c)

class SModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nz : int,
                 step : float):

        super(SModel, self).__init__()

        self.nz = nz

        self.psnet = PSNet(nc, nz)
        self.pznet = PZNet(nc, nz)
        self.qznet = QZNet(nc, nz)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.dummy_param = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    def optimize(self,
                 c : torch.Tensor,
                 s : torch.Tensor) -> tuple:
        
        self.optimizer.zero_grad()

        # sample from q(z|c,s)
        with torch.no_grad():
            z = self.qznet(c, s).sigmoid().bernoulli().detach()

        # learn p(z|c)
        pz_scores = self.pznet(c)
        pz_loss = self.criterion(pz_scores, z).sum(1).mean()
        pz_loss.backward()
        
        # learn p(s|c,z)
        ps_scores = self.psnet(c, z)
        ps_loss = self.criterion(ps_scores, s).sum([1,2,3]).mean()
        ps_loss.backward()

        # sample from p
        with torch.no_grad():
            z = self.pznet(c).sigmoid().bernoulli().detach()
            s = self.psnet(c, z).sigmoid().bernoulli().detach()

        # learn q(z|c,s)
        qz_scores = self.qznet(c, s)
        qz_loss = self.criterion(qz_scores, z).sum(1).mean()
        qz_loss.backward()

        self.optimizer.step()

        return (pz_loss.detach() + ps_loss.detach() + qz_loss.detach()) / (28 * 28 + self.nz * 2)

    def sample(self,
               c : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            z = self.pznet(c).sigmoid().bernoulli().detach()
            return self.psnet(c, z).sigmoid().bernoulli().detach()

    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        return (torch.rand([bs, 1, 28, 28], device=self.dummy_param.device) < 0.5).float().detach()

    def reconstruct(self,
                    c : torch.Tensor,
                    s : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            z = self.qznet(c, s).sigmoid().bernoulli().detach()
            return (self.psnet(c, z) > 0).float().detach()

    # this is an interface function, which is used for generation by PolyMNIST-model
    # when it is learned with synthetic examples, 
    # implements sampling p(s|c) as limiting distribution p(z,s|c)
    @torch.jit.export
    def limiting(self,
                 c : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            z = self.pznet(c).sigmoid().bernoulli().detach()
            for _ in range(100):
                s = self.psnet(c, z).sigmoid().bernoulli().detach()
                z = self.qznet(c, s).sigmoid().bernoulli().detach()
            return (self.psnet(c, z) > 0).float().detach()
