import torch

class EncoderS(torch.nn.Module):
    """ Some class description. """

    def __init__(self,
        nin : int,
        nside : int,
        nout : int,
        nn : int = 16,
        zero_init : bool = False,
        output_bias : bool = True,
        actf = torch.tanh):

        super(EncoderS, self).__init__()

        self.actf = actf

        self.conv1 = torch.nn.Conv2d(nin,     nn,      2, 1, bias=True) # 27
        self.lin1 = torch.nn.Linear(nside, nn, bias=False)
        self.conv2 = torch.nn.Conv2d(nn,      nn * 2,  3, 2, bias=True) # 13
        self.lin2 = torch.nn.Linear(nside, nn * 2, bias=False)
        self.conv3 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 11
        self.lin3 = torch.nn.Linear(nside, nn * 2, bias=False)
        self.conv4 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 9
        self.lin4 = torch.nn.Linear(nside, nn * 2, bias=False)
        self.conv5 = torch.nn.Conv2d(nn * 2,  nn * 4,  3, 2, bias=True) # 4
        self.lin5 = torch.nn.Linear(nside, nn * 4, bias=False)
        self.conv6 = torch.nn.Conv2d(nn * 4,  nout,    4, 1, bias=output_bias) # 1

        if zero_init:
            self.conv6.weight.data.zero_()
            if output_bias:
                self.conv6.bias.data.zero_()

    def forward(self,
                a : torch.Tensor,
                s : torch.Tensor) -> torch.Tensor:

        hh = self.actf(self.conv1(a) + self.lin1(s).unsqueeze(-1).unsqueeze(-1))
        hh = self.actf(self.conv2(hh) + self.lin2(s).unsqueeze(-1).unsqueeze(-1))
        hh = self.actf(self.conv3(hh) + self.lin3(s).unsqueeze(-1).unsqueeze(-1))
        hh = self.actf(self.conv4(hh) + self.lin4(s).unsqueeze(-1).unsqueeze(-1))
        hh = self.actf(self.conv5(hh) + self.lin5(s).unsqueeze(-1).unsqueeze(-1))
        return self.conv6(hh).squeeze(-1).squeeze(-1)

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
