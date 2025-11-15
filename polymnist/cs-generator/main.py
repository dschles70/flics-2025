import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import vlen, load_model, save_model
from s_model import SModel
from mnist import MNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', required=True, help='Call id.')
    parser.add_argument('--load_id', default='', help='')
    parser.add_argument('--stepsize', type=float, default=1e-4, help='Gradient step size.')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--niterations', type=int, default=-1, help='')
    parser.add_argument('--nz', type=int, required=True, help='')

    args = parser.parse_args()

    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./images', exist_ok=True)

    time0 = time.time()
    print(os.uname(), flush=True)

    # printout
    logname = './logs/log-' + args.id + '.txt'
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    nc = 10
    nz = args.nz

    # models
    s_model = SModel(nc, nz, args.stepsize).to(device)
    print('# Models prepared', file=open(logname, 'a'), flush=True)

    if args.load_id != '':
        loadprot = load_model(s_model, './models/s_' + args.load_id + '.pt')
        print('s: ', loadprot, flush=True)
        
        print('# Models loaded', file=open(logname, 'a'), flush=True)

    mnist = MNIST(args.bs)
    print('# MNIST ready', file=open(logname, 'a'), flush=True)

    log_period = 10
    save_period = 100
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    s_loss = 0.69 * torch.ones([], device=device)

    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        c, s = mnist.get_train(device)
        # add noise
        mask = (torch.rand_like(s)<0.001).float()
        s = s * (1 - mask) + (1 - s) * mask

        s_loss = s_loss * afactor1 + s_model.optimize(c, s) * afactor

        # once awhile print something out
        if count % log_period == log_period-1:

            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)
            strtoprint += ' slen: ' + str(vlen(s_model).cpu().numpy())
            strtoprint += ' sloss: ' + str(s_loss.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            save_model(s_model, './models/s_' + args.id + '.pt')

            # image
            im1 = s[0:12]
            im2 = s_model.reconstruct(c[0:12], s[0:12])
            im3 = s_model.sample(c[0:12])
            im4 = s_model.limiting(c[0:12])
            xviz = torch.cat((im1, im2, im3, im4))
            vutils.save_image(xviz, './images/img_' + args.id + '.png', nrow=12)

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
