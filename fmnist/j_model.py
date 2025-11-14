import torch
from c_model import CModel
from x_model import XModel

# the joint model consisting of two conditionals: 
# p(x|c) (conditional HVAE) and p(c|x) (FFN)
# p(c) is assumed uniform
# can be learned discriminatively (pseudo-likelihood) or by symmetric equilibrium learning

class JModel(torch.nn.Module):

    def __init__(self,
                 nz0 : int,
                 nz1 : int,
                 nn : int,
                 step : float):

        super(JModel, self).__init__()

        self.c_model = CModel(nn, step)
        self.x_model = XModel(nz0, nz1, nn, step)

        self.dummy_param = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    # def optimize(self,
    #              c_train : torch.Tensor,
    #              x_train : torch.Tensor,
    #              symmetric : bool) -> tuple[torch.Tensor, torch.Tensor]:

    #     self.c_model.optimizer.zero_grad()
    #     self.x_model.optimizer.zero_grad()
        
    #     # accumulate gradients (conditional likelihood)
    #     loss_c = self.c_model.add_grad(c_train, x_train)
    #     loss_x = self.x_model.add_grad(c_train, x_train)

    #     if symmetric: # learn generatively
    #         c_tmp = self.c_model.sample(x_train)
    #         loss_x += self.x_model.add_grad(c_tmp, x_train)
    #         x_tmp = self.x_model.sample(c_train)
    #         loss_c += self.c_model.add_grad(c_train, x_tmp)

    #     self.c_model.optimizer.step()
    #     self.x_model.optimizer.step()

    #     return loss_c, loss_x

    def optimize(self,
                 c_train : torch.Tensor,
                 x_train : torch.Tensor,
                 symmetric : bool) -> tuple[torch.Tensor, torch.Tensor]:

        if symmetric:
            c = torch.cat((c_train, c_train))
            x = torch.cat((x_train, self.x_model.sample(c_train)))
        else:
            c = c_train
            x = x_train

        loss_c = self.c_model.optimize(c, x)

        if symmetric:
            c = torch.cat((c_train, self.c_model.sample(x_train)))
            x = torch.cat((x_train, x_train))
        else:
            c = c_train
            x = x_train

        loss_x = self.x_model.optimize(c, x)

        return loss_c, loss_x

    def generate(self,
                 bs : int) -> torch.Tensor:

        c = self.c_model.sample_random(bs)
        x = self.x_model.sample(c)

        return c, x
