import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import vlen, ensure_dir
from hvae import HVAE
from mnist import MNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--call_prefix', default='tmp', help='Call prefix.')
    parser.add_argument('--load_prefix', default='', help='Load prefix.')
    parser.add_argument('--stepsize', type=float, required=True, help='Gradient step size.')
    parser.add_argument('--bs', type=int, default=100, help='Batch size')
    parser.add_argument('--niterations', type=int, default=-1, help='')
    parser.add_argument('--nz0', type=int, default=30, help='')
    parser.add_argument('--nz1', type=int, default=100, help='')
    parser.add_argument('--mode', type=int, default=0, help='')

    args = parser.parse_args()

    ensure_dir('./logs')
    ensure_dir('./models')
    ensure_dir('./images')

    time0 = time.time()
    print(os.uname(), flush=True)

    # printout
    logname = './logs/log-' + args.call_prefix + '.txt'
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    nz0 = args.nz0
    nz1 = args.nz1

    # models
    model = HVAE(nz0, nz1, args.stepsize).to(device)
    print('# Model prepared', file=open(logname, 'a'), flush=True)

    if args.load_prefix != '':
        loadprot = model.load_state_dict(torch.load('./models/mo_' + args.load_prefix + '.pt'), strict=True)
        print('model: ', loadprot, flush=True)
        print('# Model loaded', file=open(logname, 'a'), flush=True)

    mnist = MNIST()
    print('# MNIST ready', file=open(logname, 'a'), flush=True)

    log_period = 10
    save_period = 100
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    loss_q = 0.69
    loss_p = 0.69

    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        x = []

        if args.mode == 0:
            for i in range(10): # loop over the classes
                x += [mnist.get_batch(i, args.bs, device)]
        elif args.mode == 1:
            for i in range(5): # loop over the classes (first half)
                x += [mnist.get_batch(i, args.bs * 2, device)]
        else:
            for i in range(5): # loop over the classes (second half)
                x += [mnist.get_batch(i + 5, args.bs * 2, device)]

        # add noise
        x = torch.cat(x)
        mask = (torch.rand_like(x)<0.001).float()
        x = x*(1-mask) + (1-x)*mask

        llq, llp = model.optimize(x, None, None, None)
        loss_q = loss_q * afactor1 + llq * afactor
        loss_p = loss_p * afactor1 + llp * afactor

        # once awhile print something out
        if count % log_period == log_period-1:

            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' xlen: ' + str(vlen(model).cpu().numpy())
            strtoprint += ' qloss: ' + str(loss_q.cpu().numpy())
            strtoprint += ' ploss: ' + str(loss_p.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            torch.save(model.state_dict(), './models/mo_' + args.call_prefix + '.pt')

            # image
            xviz = []
            for _ in range(10):
                xviz += [model.single_shot(12)]
            xviz = torch.cat(xviz)
            vutils.save_image(xviz, './images/img_' + args.call_prefix + '.png', nrow=12)

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
