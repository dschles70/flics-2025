import torch
from nets import Decoder, Encoder, MLP

# conditional HVAE, i.e. all p and q are additionally conditioned on c

# p(z1|z0,c)
class PZ1Net(torch.nn.Module):

    def __init__(self,
                 nz0 : int,
                 nz1 : int):

        super(PZ1Net, self).__init__()

        self.net = MLP(10 + nz0, (10 + nz0 + nz1) * 4, nz1)

    def forward(self,
                c : torch.Tensor,
                z0 : torch.Tensor) -> torch.Tensor:
        
        return self.net(torch.cat((c, z0),dim=1))

# p(x|z1,c), gives means
class PXNet(torch.nn.Module):

    def __init__(self,
                 nz1 : int,
                 nn : int):

        super(PXNet, self).__init__()

        self.net = Decoder(10 + nz1, 1, nn=nn)

    def forward(self,
                c : torch.Tensor,
                z1 : torch.Tensor) -> torch.Tensor:
        
        return self.net(torch.cat((c, z1), dim=1))

# q(z1|x,c)
class QZ1Net(torch.nn.Module):

    def __init__(self,
                 nz1 : int,
                 nn : int):

        super(QZ1Net, self).__init__()

        self.net = Encoder(1, nz1, nn=nn)
        self.linc = torch.nn.Linear(10, nz1, bias=False)

    def forward(self,
                c : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:
        
        return self.linc(c) + self.net(x)

# q(z0|z1,c)
class QZ0Net(torch.nn.Module):

    def __init__(self,
                 nz0 : int,
                 nz1 : int):

        super(QZ0Net, self).__init__()
        
        self.net = MLP(10 + nz1, (10 + nz0 + nz1) * 4, nz0)

    def forward(self,
                c : torch.Tensor,
                z1 : torch.Tensor) -> torch.Tensor:
         
         return self.net(torch.cat((c, z1), dim=1))

class XModel(torch.nn.Module):

    def __init__(self,
                 nz0 : int,
                 nz1 : int,
                 nn : int,
                 step : float):

        super(XModel, self).__init__()

        self.nz0 = nz0

        self.pz1net = PZ1Net(nz0, nz1)
        self.pxnet = PXNet(nz1, nn)
        self.qz1net = QZ1Net(nz1, nn)
        self.qz0net = QZ0Net(nz0, nz1)

        # one global learnable sigma
        self.lsigma = torch.nn.Parameter(torch.zeros([]), requires_grad=True)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.dummy = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    # summetric learning, conditioned on c
    def optimize(self,
                 c : torch.Tensor,
                 x : torch.Tensor) -> torch.Tensor:

        self.optimizer.zero_grad()

        # sample from q
        with torch.no_grad():
            qz1 = self.qz1net(c, x).sigmoid().bernoulli().detach()
            qz0 = self.qz0net(c, qz1).sigmoid().bernoulli().detach()

        # optimize p
        pz1_scores = self.pz1net(c, qz0)
        pz1_loss = self.criterion(pz1_scores, qz1).sum(1).mean()
        px_mu = self.pxnet(c, qz1)
        sigma = torch.exp(self.lsigma)
        px_loss = 0.91894 + self.lsigma + ((x - px_mu)**2) / (2*(sigma**2))
        px_loss = px_loss.sum([1,2,3]).mean()

        # sample from p
        with torch.no_grad():
            pz0 = (torch.rand_like(qz0)>0.5).float()
            pz1 = self.pz1net(c, pz0).sigmoid().bernoulli().detach()
            px_mu = self.pxnet(c, pz1)
            px = (torch.randn_like(px_mu)*sigma + px_mu).detach()
        
        # optimize q
        qz1_scores = self.qz1net(c, px)
        qz1_loss = self.criterion(qz1_scores, pz1).sum(1).mean() 
        qz0_scores = self.qz0net(c, pz1)
        qz0_loss = self.criterion(qz0_scores, pz0).sum(1).mean() 

        loss = pz1_loss + px_loss + qz1_loss + qz0_loss

        loss.backward()

        self.optimizer.step()

        nz0 = pz0.shape[0]
        nz1 = pz1.shape[0]

        return loss.detach()/(28*28 + nz0 + nz1 * 2)

    # for vizualization -> gives image means
    def shot(self,
             c : torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            z0 = (torch.rand([c.shape[0], self.nz0], device=self.dummy.device)>0.5).float()
            z1 = self.pz1net(c, z0).sigmoid().bernoulli().detach()
            return self.pxnet(c, z1).detach()
        
    def sample(self,
               c : torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            mu = self.shot(c)
            sigma = torch.exp(self.lsigma)
            return (torch.randn_like(mu)*sigma + mu).detach()
