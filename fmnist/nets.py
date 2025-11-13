import torch

# 2d -> 1d
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

# 2d -> 1d
class Decoder(torch.nn.Module):
    """ Some class description. """

    def __init__(self,
        nin : int,
        nout : int,
        nn : int = 16,
        zero_init : bool = False,
        output_bias : bool = True,
        actf = torch.tanh):

        super(Decoder, self).__init__()

        self.actf = actf
        
        self.conv1 = torch.nn.ConvTranspose2d(nin,    nn * 4, 4, 1, bias=True) # 4
        self.conv2 = torch.nn.ConvTranspose2d(nn * 4, nn * 2, 3, 2, bias=True) # 9
        self.conv3 = torch.nn.ConvTranspose2d(nn * 2, nn * 2, 3, 1, bias=True) # 11
        self.conv4 = torch.nn.ConvTranspose2d(nn * 2, nn * 2, 3, 1, bias=True) # 13
        self.conv5 = torch.nn.ConvTranspose2d(nn * 2, nn,     3, 2, bias=True) # 27
        self.conv6 = torch.nn.ConvTranspose2d(nn,     nout,   2, 1, bias=output_bias) # 28

        if zero_init:
            self.conv6.weight.data.zero_()
            if output_bias:
                self.conv6.bias.data.zero_()

    def forward(self,
                a : torch.Tensor) -> torch.Tensor:

        hh = self.actf(self.conv1(a.unsqueeze(-1).unsqueeze(-1)))
        hh = self.actf(self.conv2(hh))
        hh = self.actf(self.conv3(hh))
        hh = self.actf(self.conv4(hh))
        hh = self.actf(self.conv5(hh))
        return self.conv6(hh)

# MLP network
class MLP(torch.nn.Module):
    """ Some class description. """

    def __init__(self,
        nin : int,
        nn : int,
        nout : int,
        nhlayers : int = 2,
        zero_init : bool = False,
        output_bias : bool = True,
        actf = torch.tanh):

        super(MLP, self).__init__()

        self.actf = actf

        self.nhlayers = nhlayers
        nprev = nin
        for i in range(self.nhlayers):
            ll = torch.nn.Linear(nprev, nn, bias=True)
            self.add_module('lin_%d' % i, ll)
            nprev = nn
        self.lin_last = torch.nn.Linear(nprev, nout, bias=output_bias)
        if zero_init:
            self.lin_last.weight.data.zero_()
            if output_bias:
                self.lin_last.bias.data.zero_()

    def forward(self, a : torch.Tensor) -> torch.Tensor:
        hh = a
        for i in range(self.nhlayers):
            mm = getattr(self, 'lin_%d' % i)
            hh = self.actf(mm(hh))
        return self.lin_last(hh)
