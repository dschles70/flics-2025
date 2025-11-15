import os
import argparse
import time
import torch

from helpers import load_model

from c_model import CModel

from polymnist import POLYMNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', required=True, help='Call prefix.')

    args = parser.parse_args()

    time0 = time.time()
    print(os.uname(), flush=True)

    print('Starting ...', flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    nc = 10
    nd = 5

    polymnist = POLYMNIST(100)
    print('POLYMNIST ready', flush=True)

    # models
    c_model = CModel(nc, 0).to(device)
    
    loadprot = load_model(c_model, './models/c_' + args.id + '.pt')
    print('c: ', loadprot, flush=True)

    print('Models loaded', flush=True)

    ###### Validation set #####

    x = torch.split(polymnist.x_val, 1000)
    c = torch.split(polymnist.c_val, 1000)
    d = torch.split(polymnist.d_val, 1000)
    n_batches = len(x)
    print(n_batches, 'validation batches found')

    c_acc = [0] * nd
    n = [0] * nd

    for i in range(n_batches):
        cc = c[i].to(device)
        dd = d[i].to(device)
        xx = x[i].to(device)

        c_gt = torch.argmax(cc, dim=1)
        d_gt = torch.argmax(dd, dim=1)
        with torch.no_grad():
            c_scores = c_model.net(xx)
            c_dec = torch.argmax(c_scores, dim=1)

        for j in range(nd):
            c_gt_tmp = c_gt[d_gt == j]
            c_dec_tmp = c_dec[d_gt == j]
            c_acc[j] = c_acc[j] + (c_gt_tmp==c_dec_tmp).float().sum()
            n[j] = n[j] + c_gt_tmp.shape[0]

        print('.', end='', flush=True)

    avg_acc = sum(c_acc) / sum(n)

    for j in range(nd):
        c_acc[j] = (c_acc[j] / n[j]).cpu().numpy()

    print()
    print('Classification accuracy (val)', c_acc, avg_acc.cpu().numpy(), flush=True)
