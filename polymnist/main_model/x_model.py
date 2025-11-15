import torch
from nets import EncoderS, Decoder

def cal(y):
    with torch.no_grad():
        mean = y.mean(0, keepdim=True)
        y_norm = y - mean
        std = torch.sqrt((y_norm ** 2).mean(0, keepdim=True))
        y_norm = y_norm / std

    return (y - y_norm) ** 2

class XPrior(torch.nn.Module):
    def __init__(self,
                 nc : int,
                 nd : int,
                 ny : int,
                 nz : int):

        super(XPrior, self).__init__()

        nn = (nc + nd + nz + ny * 2) * 2
        self.lin1 = torch.nn.Linear(nc + nd + nz, nn)
        self.lin2 = torch.nn.Linear(nn, nn)
        self.lin3 = torch.nn.Linear(nn, ny * 2)

    def forward(self,
                c : torch.Tensor,
                d : torch.Tensor,
                z : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
         
         hh = torch.cat((c, d, z), dim=1)
         hh = torch.tanh(self.lin1(hh))
         hh = torch.tanh(self.lin2(hh))
         scores = self.lin3(hh)
         mu, lsigma = torch.chunk(scores, 2, dim=1)
         return mu, lsigma

class XEncoder(torch.nn.Module):
    def __init__(self,
                 nc : int,
                 nd : int,
                 ny : int,
                 nz : int,
                 iconst : torch.Tensor):

        super(XEncoder, self).__init__()

        self.net = EncoderS(3, nc + nd + nz, ny * 2)
        self.iconst = iconst

    def forward(self,
                c : torch.Tensor,
                d : torch.Tensor,
                z : torch.Tensor,
                s : torch.Tensor,
                x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
         
         x = x * (1 - s) + (self.iconst - x) * s
         cdz = torch.cat((c, d, z), dim=1)
         scores = self.net(x, cdz)
         mu, lsigma = torch.chunk(scores, 2, dim=1)
         return mu, lsigma

class XDecoder(torch.nn.Module):
    def __init__(self,
                 nc : int,
                 nd : int,
                 ny : int,
                 nz : int,
                 iconst : torch.Tensor):

        super(XDecoder, self).__init__()

        self.net = Decoder(nc + nd + ny + nz, 6)
        self.iconst = iconst

    def forward(self,
                c : torch.Tensor,
                d : torch.Tensor,
                y : torch.Tensor,
                z : torch.Tensor,
                s : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
         
         cdyz = torch.cat((c, d, y, z), dim=1)
         scores = self.net(cdyz)
         mu, lsigma = torch.chunk(scores, 2, dim=1)
         mu = mu * (1 - s) + (self.iconst - mu) * s

         return mu, lsigma

class XModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 ny : int,
                 nz : int,
                 step : float,
                 iconst : torch.Tensor):

        super(XModel, self).__init__()

        self.ny = ny

        self.encoder = XEncoder(nc, nd, ny, nz, iconst)
        self.decoder = XDecoder(nc, nd, ny, nz, iconst)
        self.prior = XPrior(nc, nd, ny, nz)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        self.sheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=(lambda i: (100000 / (100000 + i))) )

        self.dummy_param = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    def optimize(self,
                 c : torch.Tensor,
                 d : torch.Tensor,
                 z : torch.Tensor,
                 s : torch.Tensor,
                 x : torch.Tensor,
                 l : torch.Tensor) -> tuple:

        c = c[l!=2]
        d = d[l!=2]
        z = z[l!=2]
        s = s[l!=2]
        x = x[l!=2]
        
        self.optimizer.zero_grad()

        ymu_p, ylsigma_p = self.prior(c, d, z)
        ysigma_p = torch.exp(ylsigma_p)

        ymu, ylsigma = self.encoder(c, d, z, s, x)
        ysigma = torch.exp(ylsigma)
        y = torch.randn_like(ymu) * ysigma + ymu
        xmu, xlsigma = self.decoder(c, d, y, z, s)
        xsigma = torch.exp(xlsigma)

        # data loss
        xloss = 0.91894 + xlsigma + ((x - xmu)**2) / (2*(xsigma**2))
        xloss = xloss.sum([1,2,3]).mean()

        # kl-loss
        ydistr_p = torch.distributions.normal.Normal(ymu_p, ysigma_p)
        ydistr_q = torch.distributions.normal.Normal(ymu, ysigma)
        klloss = torch.distributions.kl_divergence(ydistr_q, ydistr_p).sum(1).mean()

        # calibration loss
        calloss = cal(y).sum(1).mean()

        loss = xloss + klloss + calloss
        loss.backward()

        self.optimizer.step()
        self.sheduler.step()

        return loss.detach() / (28 * 28 * 3 + self.ny)

    def sample(self,
               c : torch.Tensor,
               d : torch.Tensor,
               z : torch.Tensor,
               s : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            mu, lsigma = self.prior(c, d, z)
            sigma = torch.exp(lsigma)
            y = torch.randn_like(mu)*sigma + mu
            mu, lsigma = self.decoder(c, d, y, z, s)
            sigma = torch.exp(lsigma)
            x = torch.randn_like(mu)*sigma + mu
            return x.detach()

    def get_ms(self,
               c : torch.Tensor,
               d : torch.Tensor,
               z : torch.Tensor,
               s : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            mu, lsigma = self.prior(c, d, z)
            sigma = torch.exp(lsigma)
            y = torch.randn_like(mu)*sigma + mu
            mu, lsigma = self.decoder(c, d, y, z, s)
            sigma = torch.exp(lsigma)
            return mu.detach(), sigma.detach()

    def reconstruct(self,
                    c : torch.Tensor,
                    d : torch.Tensor,
                    z : torch.Tensor,
                    s : torch.Tensor,
                    x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            ymu, ylsigma = self.encoder(c, d, z, s, x)
            ysigma = torch.exp(ylsigma)
            y = torch.randn_like(ymu) * ysigma + ymu
            mu, _ = self.decoder(c, d, y, z, s)
            return mu.detach()
