import os
import argparse
import time
import torch
import torchvision.utils as vutils

from hvae import HVAE
from mnist import MNIST

def vlen(layer : torch.nn.Module) -> float:
    vsum = 0.
    vnn = 0
    for vv in layer.parameters():
        if vv.requires_grad:
            param = vv.data
            vsum = vsum + (param*param).sum()
            vnn = vnn + param.numel()
    return vsum/vnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', default='tmp', help='Call identifier.')
    parser.add_argument('--load_id', default='', help='Load identifier.')
    parser.add_argument('--stepsize', type=float, required=True, help='Gradient step size.')
    parser.add_argument('--niterations', type=int, default=-1, help='')
    parser.add_argument('--nz0', type=int, default=30, help='')
    parser.add_argument('--nz1', type=int, default=100, help='')
    parser.add_argument('--alpha', type=float, required=True, help='')

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

    nz0 = args.nz0
    nz1 = args.nz1

    # two HVAE models: for 0...4 and 5...9 (see the paper)
    model0 = HVAE(nz0, nz1, args.stepsize).to(device)
    model1 = HVAE(nz0, nz1, args.stepsize).to(device)
    print('# Models prepared', file=open(logname, 'a'), flush=True)

    if args.load_id != '':
        loadprot = model0.load_state_dict(torch.load('./models/mt0_' + args.load_id + '.pt'), strict=True)
        print('model0: ', loadprot, flush=True)
        loadprot = model1.load_state_dict(torch.load('./models/mt1_' + args.load_id + '.pt'), strict=True)
        print('model1: ', loadprot, flush=True)

        print('# Models loaded', file=open(logname, 'a'), flush=True)

    mnist = MNIST()
    print('# MNIST ready', file=open(logname, 'a'), flush=True)

    log_period = 10
    save_period = 100
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    loss_q = 0.69 # ln(2) -- the "worst" bynary cross-entropy loss
    loss_p = 0.69

    while True:

        # how many samples of each kind (own per digits / generated)
        n_own = int(200 * args.alpha)
        n_gen = int(1000 - n_own * 5)

        # prepare generated data
        z0_g0, z1_g0, x_g0 = model0.sample_p(n_gen)
        z0_g1, z1_g1, x_g1 = model1.sample_p(n_gen)
        
        # get real data
        x_own0 = []
        x_own1 = []
        for i in range(5):
            x_own0 += [mnist.get_batch(i, n_own, device)]
            x_own1 += [mnist.get_batch(i + 5, n_own, device)]
        x_own0 = torch.cat(x_own0)
        x_own1 = torch.cat(x_own1)
        
        # add small noise (salt and pepper) to the real data to avoid zero probabilities
        mask = (torch.rand_like(x_own0)<0.001).float()
        x_own0 = x_own0*(1-mask) + (1-x_own0)*mask

        mask = (torch.rand_like(x_own1)<0.001).float()
        x_own1 = x_own1*(1-mask) + (1-x_own1)*mask

        # learn HVAEs
        llq0, llp0 = model0.optimize(x_own0, z0_g1, z1_g1, x_g1)
        llq1, llp1 = model1.optimize(x_own1, z0_g0, z1_g0, x_g0)

        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        # accumulate losses
        loss_q = loss_q * afactor1 + (llq0 + llq1) * afactor / 2
        loss_p = loss_p * afactor1 + (llp0 + llp1) * afactor / 2

        # once awhile print something out
        if count % log_period == log_period-1:

            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' xlen: ' + str((vlen(model0).cpu().numpy() + vlen(model1).cpu().numpy())/2)
            strtoprint += ' qloss: ' + str(loss_q.cpu().numpy())
            strtoprint += ' ploss: ' + str(loss_p.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            torch.save(model0.state_dict(), './models/mt0_' + args.id + '.pt')
            torch.save(model1.state_dict(), './models/mt1_' + args.id + '.pt')

            # image
            xviz0 = model0.single_shot(60)
            xviz1 = model1.single_shot(60)
            xviz = torch.cat((xviz0, xviz1))
            vutils.save_image(xviz, './images/img_' + args.id + '.png', nrow=12)

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == niterations:
            break

    print('# Final models saving ...', file=open(logname, 'a'), flush=True)
    torch.save(model0.state_dict(), './models/mt0_' + args.id + '.pt')
    torch.save(model1.state_dict(), './models/mt1_' + args.id + '.pt')

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
