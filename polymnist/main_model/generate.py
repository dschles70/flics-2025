import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import load_model

from z_model import ZModel
from s_model import SModel
from x_model import XModel
from polymnist import POLYMNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--call_prefix', required=True, help='Call prefix.')
    parser.add_argument('--ny', type=int, required=True, help='')
    parser.add_argument('--nz', type=int, required=True, help='')

    args = parser.parse_args()

    os.makedirs('./images', exist_ok=True)

    time0 = time.time()
    print(os.uname(), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    nc = 10
    nd = 5
    ny = args.ny
    nz = args.nz

    polymnist = POLYMNIST(100)
    print('POLYMNIST ready', flush=True)

    means = polymnist.means.to(device)
    std = polymnist.std.to(device)
    iconst = (1 - 2 * means) / std

    # models
    z_model = ZModel(nc, nd, nz, 0).to(device)
    s_model = SModel(nc, nd, nz, 0).to(device)
    x_model = XModel(nc, nd, ny, nz, 0, iconst).to(device)
    
    loadprot = load_model(z_model, './models/z_' + args.call_prefix + '.pt')
    print('z: ', loadprot, flush=True)
    loadprot = load_model(s_model, './models/s_' + args.call_prefix + '.pt')
    print('s: ', loadprot, flush=True)
    loadprot = load_model(x_model, './models/x_' + args.call_prefix + '.pt')
    print('x: ', loadprot, flush=True)
    print('Everything prepared, go ...', flush=True)

    c = torch.zeros([50]).long()
    d = torch.zeros([50]).long()
    for i in range(50):
        c[i] = i % 10
        d[i] = i // 10
    c = torch.nn.functional.one_hot(c, num_classes=10).float().to(device)
    d = torch.nn.functional.one_hot(d, num_classes=5).float().to(device)

    z = z_model.sample_random(50)
    s = s_model.sample_random(50)
    x = x_model.sample(c, d, z, s)
    
    for _ in range(1000):
        for op in torch.randperm(3):
            if op == 0:
                z = z_model.sample(c, d, s, x)
            elif op == 1:
                s = s_model.sample(c, d, z, x)
            else:
                x = x_model.sample(c, d, z, s)
                # x = x_model.reconstruct(c, d, z, s, x)

    s = s_model.get_dec(c, d, z, x)
    x = x_model.get_ms(c, d, z, s)[0] * std + means
    # x = x_model.reconstruct(c, d, z, s, x) * std + means
    vutils.save_image(x, './images/generated_' + args.call_prefix + '.png', nrow=10)

    print('Done.', flush=True)
