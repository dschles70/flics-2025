import torch

class Encoder(torch.nn.Module):
    """ Some class description. """

    def __init__(self,
        nin : int,
        nout : int,
        nn : int = 16,
        zero_init : bool = False,
        output_bias : bool = True,
        actf = torch.tanh):

        super(Encoder, self).__init__()

        self.actf = actf

        self.conv1 = torch.nn.Conv2d(nin,     nn,      2, 1, bias=True) # 27
        self.conv2 = torch.nn.Conv2d(nn,      nn * 2,  3, 2, bias=True) # 13
        self.conv3 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 11
        self.conv4 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 9
        self.conv5 = torch.nn.Conv2d(nn * 2,  nn * 4,  3, 2, bias=True) # 4
        self.conv6 = torch.nn.Conv2d(nn * 4,  nout,    4, 1, bias=output_bias) # 1

        if zero_init:
            self.conv6.weight.data.zero_()
            if output_bias:
                self.conv6.bias.data.zero_()

    def forward(self,
                a : torch.Tensor) -> torch.Tensor:

        hh = self.actf(self.conv1(a))
        hh = self.actf(self.conv2(hh))
        hh = self.actf(self.conv3(hh))
        hh = self.actf(self.conv4(hh))
        hh = self.actf(self.conv5(hh))
        return self.conv6(hh).squeeze(-1).squeeze(-1)

class CModel(torch.nn.Module):

    def __init__(self,
                 nc : int,
                 step : float):

        super(CModel, self).__init__()

        self.nc = nc

        self.net = Encoder(3, nc)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)
        # self.sheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=(lambda i: (100000 / (100000 + i))) )
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.dummy_param = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    def optimize(self,
                 c : torch.Tensor,
                 x : torch.Tensor) -> tuple:
        
        self.optimizer.zero_grad()

        scores = self.net(x)
        loss = self.criterion(scores, c).mean()
        loss.backward()

        self.optimizer.step()
        # self.sheduler.step()

        return loss.detach()
