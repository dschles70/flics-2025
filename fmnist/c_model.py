import torch
from nets import Encoder

# p(c|x), just an encoder
class CModel(torch.nn.Module):

    def __init__(self,
                 nn : int,
                 step : float):

        super(CModel, self).__init__()

        self.net = Encoder(1, 10, nn = nn)
        
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step)

        self.dummy = torch.nn.Parameter(torch.zeros([]), requires_grad=False)

    def optimize(self,
                 c : torch.Tensor,
                 x : torch.Tensor) -> torch.Tensor:

        self.optimizer.zero_grad()

        scores = self.net(x)
        cind = torch.argmax(c, dim=1)
        loss = self.criterion(scores, cind).mean()
        loss.backward()

        self.optimizer.step()

        return loss.detach()

    def sample(self,
               x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scores = self.net(x)
            return torch.distributions.one_hot_categorical.OneHotCategorical(logits=scores).sample().float().detach()

    def sample_random(self,
                      bs : int) -> torch.Tensor:
        
        cind = torch.randint(10, [bs], device=self.dummy.device)
        return torch.nn.functional.one_hot(cind, num_classes=10).float()
