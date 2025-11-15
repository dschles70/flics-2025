import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import load_model

from d_model import DModel
from z_model import ZModel
from x_model import XModel
from polymnist import POLYMNIST
from cs_source import CSSOURCE

from main import complete_cds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--call_prefix', required=True, help='Call prefix.')
    parser.add_argument('--ny', type=int, required=True, help='')
    parser.add_argument('--nz', type=int, required=True, help='')
    parser.add_argument('--generator_path', default='', help='')

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

    cs_source = CSSOURCE(1000, args.generator_path, device)
    print('CSSource ready', flush=True)

    polymnist = POLYMNIST(1000)
    print('POLYMNIST ready', flush=True)

    means = polymnist.means.to(device)
    std = polymnist.std.to(device)
    iconst = (1 - 2 * means) / std

    # models
    d_model = DModel(nc, nd, nz, 0, iconst).to(device)
    z_model = ZModel(nc, nd, nz, 0).to(device)
    x_model = XModel(nc, nd, ny, nz, 0, iconst).to(device)
    
    loadprot = load_model(d_model, './models/d_' + args.call_prefix + '.pt')
    print('d: ', loadprot, flush=True)
    loadprot = load_model(z_model, './models/z_' + args.call_prefix + '.pt')
    print('z: ', loadprot, flush=True)
    loadprot = load_model(x_model, './models/x_' + args.call_prefix + '.pt')
    print('x: ', loadprot, flush=True)
    print('Everything prepared, go ...', flush=True)

    c, s = cs_source.get_train(device)
    d = d_model.sample_random(1000)
    z, x, _ = complete_cds(c, d, s, z_model, x_model)
    x = x_model.get_ms(c, d, z, s)[0] * std + means

    # sort out
    x50 = torch.zeros([50,3,28,28], device=device)
    for i in range(1000):
        cc = torch.argmax(c[i])
        dd = torch.argmax(d[i])
        x50[dd * 10 + cc] = x[i]

    vutils.save_image(x50, './images/generated1_' + args.call_prefix + '.png', nrow=10)

    # originals
    c, d, x = polymnist.get_train(device)
    x = x * std + means

    x50 = torch.zeros([50,3,28,28], device=device)
    for i in range(1000):
        cc = torch.argmax(c[i])
        dd = torch.argmax(d[i])
        x50[dd * 10 + cc] = x[i]

    vutils.save_image(x50, './images/original.png', nrow=10)

    print('Done.', flush=True)
