import torch

# q(z1|x)
class QNETx1(torch.nn.Module):
    def __init__(self,
                 nz1 : int,
                 nn : int = 32):
        
        super(QNETx1, self).__init__()

        self.actf = torch.tanh

        # 3*28*28 ->
        self.conv1 = torch.nn.Conv2d(1,       nn,      2, 1, bias=True) # 27
        self.conv2 = torch.nn.Conv2d(nn,      nn * 2,  3, 2, bias=True) # 13
        self.conv3 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 11
        self.conv4 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 9
        self.conv5 = torch.nn.Conv2d(nn * 2,  nn * 4,  3, 2, bias=True) # 4
        self.conv6 = torch.nn.Conv2d(nn * 4,  nz1,     4, 1, bias=True) # 1

    def forward(self,
                x : torch.Tensor) -> torch.Tensor:

        hh = self.actf(self.conv1(x))
        hh = self.actf(self.conv2(hh))
        hh = self.actf(self.conv3(hh))
        hh = self.actf(self.conv4(hh))
        hh = self.actf(self.conv5(hh))
        return self.conv6(hh).squeeze(-1).squeeze(-1)

# q(z0|z1,x)
class QNET10(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int):
        
        super(QNET10, self).__init__()

        self.actf = torch.tanh
        nn = 32

        # 3*28*28 ->
        self.conv1 = torch.nn.Conv2d(1,       nn,      2, 1, bias=True) # 27
        self.lin1 = torch.nn.Linear(nz1, nn, bias=False)
        self.conv2 = torch.nn.Conv2d(nn,      nn * 2,  3, 2, bias=True) # 13
        self.lin2 = torch.nn.Linear(nz1, nn * 2, bias=False)
        self.conv3 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 11
        self.lin3 = torch.nn.Linear(nz1, nn * 2, bias=False)
        self.conv4 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 9
        self.lin4 = torch.nn.Linear(nz1, nn * 2, bias=False)
        self.conv5 = torch.nn.Conv2d(nn * 2,  nn * 4,  3, 2, bias=True) # 4
        self.lin5 = torch.nn.Linear(nz1, nn * 4, bias=False)
        self.conv6 = torch.nn.Conv2d(nn * 4,  nz0,  4, 1, bias=True) # 1

    def forward(self,
                z1 : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:

        hh = self.actf(self.conv1(x) + self.lin1(z1).unsqueeze(-1).unsqueeze(-1))
        hh = self.actf(self.conv2(hh) + self.lin2(z1).unsqueeze(-1).unsqueeze(-1))
        hh = self.actf(self.conv3(hh) + self.lin3(z1).unsqueeze(-1).unsqueeze(-1))
        hh = self.actf(self.conv4(hh) + self.lin4(z1).unsqueeze(-1).unsqueeze(-1))
        hh = self.actf(self.conv5(hh) + self.lin5(z1).unsqueeze(-1).unsqueeze(-1))
        return self.conv6(hh).squeeze(-1).squeeze(-1)

# p(z1|z0)
class PNET01(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int):
        
        super(PNET01, self).__init__()

        self.actf = torch.tanh

        nnmlp = 600

        self.lin1 = torch.nn.Linear(nz0,   nnmlp, bias=True)
        self.lin2 = torch.nn.Linear(nnmlp, nnmlp, bias=True)
        self.lin3 = torch.nn.Linear(nnmlp, nz1,   bias=True)

        self.lin_skip = torch.nn.Linear(nz0, nz1, bias=False)

    def forward(self,
                z0 : torch.Tensor) -> torch.Tensor:
        
        hh = self.actf(self.lin1(z0))
        hh = self.actf(self.lin2(hh))
        return self.lin3(hh) + self.lin_skip(z0)

# p(x|z0,z1)
class PNET1x(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int,
                 nn : int = 32):
        
        super(PNET1x, self).__init__()

        self.actf = torch.tanh

        self.conv1 = torch.nn.ConvTranspose2d(nz0 + nz1,    nn * 4, 4, 1, bias=True) # 4
        self.conv2 = torch.nn.ConvTranspose2d(nn * 4, nn * 2, 3, 2, bias=True) # 9
        self.conv3 = torch.nn.ConvTranspose2d(nn * 2, nn * 2, 3, 1, bias=True) # 11
        self.conv4 = torch.nn.ConvTranspose2d(nn * 2, nn * 2, 3, 1, bias=True) # 13
        self.conv5 = torch.nn.ConvTranspose2d(nn * 2, nn,     3, 2, bias=True) # 27
        self.conv6 = torch.nn.ConvTranspose2d(nn,     1,     2, 1, bias=True) # 28

    def forward(self,
                z0 : torch.Tensor,
                z1 : torch.Tensor) -> torch.Tensor:

        hh = torch.cat((z0, z1), dim=1).unsqueeze(-1).unsqueeze(-1)

        hh = self.actf(self.conv1(hh))
        hh = self.actf(self.conv2(hh))
        hh = self.actf(self.conv3(hh))
        hh = self.actf(self.conv4(hh))
        hh = self.actf(self.conv5(hh))
        return self.conv6(hh)

class HVAE(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int,
                 step : float):
        
        super(HVAE, self).__init__()

        self.nz0 = nz0

        self.pnet01 = PNET01(nz0, nz1)
        self.pnet1x = PNET1x(nz0, nz1)
        self.qnetx1 = QNETx1(nz1)
        self.qnet10 = QNET10(nz0, nz1)

        self.optimizer_p = torch.optim.Adam([
            {'params': self.pnet01.parameters()},
            {'params': self.pnet1x.parameters()}
        ], lr=step)

        self.optimizer_q = torch.optim.Adam([
            {'params': self.qnet10.parameters()},
            {'params': self.qnetx1.parameters()}
        ], lr=step)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        # just to know the device, we are working on (see the usage below)
        self.dummy_param = torch.nn.Parameter(torch.zeros([]))

    # sampling from p (decoder)
    def sample_p(self,
                 bs : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            z0 = (torch.rand([bs, self.nz0], device = self.dummy_param.device)>0.5).float()
            z1 = self.pnet01(z0).sigmoid().bernoulli().detach()
            x = self.pnet1x(z0, z1).sigmoid().bernoulli().detach()
            return z0, z1, x

    # sampling from q (encoder)
    def sample_q(self,
                 x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            z1 = self.qnetx1(x).sigmoid().bernoulli().detach()
            z0 = self.qnet10(z1, x).sigmoid().bernoulli().detach()
            return z0, z1

    def optimize_p(self, 
                   z0_gt : torch.Tensor,
                   z1_gt : torch.Tensor,
                   x_gt : torch.Tensor) -> torch.Tensor:
        
        self.optimizer_p.zero_grad()

        z1_scores = self.pnet01(z0_gt)
        loss_z1 = self.criterion(z1_scores, z1_gt).sum(1).mean()

        x_scores = self.pnet1x(z0_gt, z1_gt)
        loss_x = self.criterion(x_scores, x_gt).sum([1,2,3]).mean()

        loss = loss_z1 + loss_x
        loss.backward()
        self.optimizer_p.step()
        return loss.detach()/(28*28 + z1_gt.shape[1])

    def optimize_q(self,
                   z0_gt : torch.Tensor,
                   z1_gt : torch.Tensor,
                   x_gt : torch.Tensor) -> torch.Tensor:
        
        self.optimizer_q.zero_grad()

        z1_scores = self.qnetx1(x_gt)
        loss_z1 = self.criterion(z1_scores, z1_gt).sum(1).mean()

        z0_scores = self.qnet10(z1_gt, x_gt)
        loss_z0 = self.criterion(z0_scores, z0_gt).sum(1).mean()

        loss = loss_z0 + loss_z1
        loss.backward()
        self.optimizer_q.step()
        return loss.detach()/(z0_gt.shape[1] + z1_gt.shape[1])
    
    def optimize(self,
                 x : torch.Tensor, # unsupervized (i.e. z0,z1 are not given) own real data
                 z0_o : torch.Tensor, # fully supervized (z0,z1,x) synthetic data, None if no synthetic data
                 z1_o : torch.Tensor,
                 x_o : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Symmetric equilibrium learning:

        # 1) sample from q (add synthetic supervized if any)
        z0, z1 = self.sample_q(x)
        if z0_o is None:
            z0l = z0
            z1l = z1
            xl = x
        else:
            z0l = torch.cat((z0, z0_o))
            z1l = torch.cat((z1, z1_o))
            xl = torch.cat((x, x_o))
        
        # 2) optimize p
        loss_p = self.optimize_p(z0l, z1l, xl)

        # 3) sample from p (add synthetic supervized if any)
        z0, z1, x = self.sample_p(z0.shape[0])
        if z0_o is None:
            z0l = z0
            z1l = z1
            xl = x
        else:
            z0l = torch.cat((z0, z0_o))
            z1l = torch.cat((z1, z1_o))
            xl = torch.cat((x, x_o))

        # 4) optimize q
        loss_q = self.optimize_q(z0l, z1l, xl)

        return loss_q.detach(), loss_p.detach()        

    # just for vizualization, basically the same as "sample_p" but give only x-probabilities back
    def single_shot(self,
                    bs : int) -> torch.Tensor:
        
        with torch.no_grad():
            z0 = (torch.rand([bs, self.nz0], device = self.dummy_param.device)>0.5).float()
            z1 = self.pnet01(z0).sigmoid().bernoulli().detach()
            return self.pnet1x(z0, z1).sigmoid().detach()
