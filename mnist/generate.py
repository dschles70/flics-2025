import torch
import torchvision.utils as vutils

from hvae import HVAE

# entry point, the main
if __name__ == '__main__':

    # printout
    print('# Start ...', flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    nz0 = 30
    nz1 = 100

    # the model
    model = HVAE(nz0, nz1, 0).to(device)
    loadprot = model.load_state_dict(torch.load('./models/mo_20.pt'), strict=True)
    x1 = model.single_shot(18).float()
    loadprot = model.load_state_dict(torch.load('./models/mt0_19.pt'), strict=True)
    x2 = model.single_shot(18).float()
    loadprot = model.load_state_dict(torch.load('./models/mt1_19.pt'), strict=True)
    x3 = model.single_shot(18).float()
    loadprot = model.load_state_dict(torch.load('./models/mt0_10.pt'), strict=True)
    x4 = model.single_shot(18).float()
    loadprot = model.load_state_dict(torch.load('./models/mt1_10.pt'), strict=True)
    x5 = model.single_shot(18).float()

    x = torch.cat((x1, x2, x3, x4, x5))
    vutils.save_image(x, './images/generated.png', nrow=18)

    print(' done.')
