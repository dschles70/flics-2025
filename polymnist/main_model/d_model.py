import torch
from nets import Encoder

# We experimented here with different architectures,
# that's why some input parameters are sometimes redundant

class DNet(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 nz : int,
                 iconst : torch.Tensor):

        super(DNet, self).__init__()

        self.net = Encoder(3, nd)
        self.iconst = iconst

    def forward(self,
                c : torch.Tensor,
                z : torch.Tensor,
                s : torch.Tensor,
                x : torch.Tensor) -> torch.Tensor:
         
         xx = x * (1 - s) + (self.iconst - x) * s
         return self.net(xx)

class DModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 nd : int,
                 nz : int,
                 step : float,
                 iconst : torch.Tensor):

        super(DModel, self).__init__()

        self.nd = nd

        self.net = DNet(nc, nd, nz, iconst)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        # self.sheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=(lambda i: (100000 / (100000 + i))) )
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.dummy_param = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    def optimize(self,
                 c : torch.Tensor,
                 d : torch.Tensor,
                 z : torch.Tensor,
                 s : torch.Tensor,
                 x : torch.Tensor) -> tuple:
        
        self.optimizer.zero_grad()

        scores = self.net(c, z, s, x)
        loss = self.criterion(scores, d).mean()
        loss.backward()

        self.optimizer.step()
        # self.sheduler.step()

        return loss.detach()

    def sample(self,
               c : torch.Tensor,
               z : torch.Tensor,
               s : torch.Tensor,
               x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(c, z, s, x)
            distr = torch.distributions.one_hot_categorical.OneHotCategorical(logits=scores)
            return distr.sample()

    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        ind = torch.randint(self.nd, [bs], device=self.dummy_param.device)
        return torch.nn.functional.one_hot(ind, num_classes=self.nd).float()
