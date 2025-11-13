import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import *
from j_model import JModel
from fmnist import FMNIST

def inference(
        x : torch.Tensor,
        j_model : JModel) -> float:
    
    with torch.no_grad():
        c_scores = j_model.c_model.net(x)
        cind = torch.argmax(c_scores, dim=1)
        return torch.nn.functional.one_hot(cind, num_classes=10).float()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', default='tmp', help='')
    parser.add_argument('--load_id', default='', help='')
    parser.add_argument('--stepsize', type=float, required=True, help='')
    parser.add_argument('--niterations', type=int, default=-1, help='')
    parser.add_argument('--bs', type=int, required=True, help='')

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

    n_models = 4
    nz0 = [4, 8,  12, 16]
    nz1 = [8, 16, 24, 32]
    nn =  [2, 4,  8,  16]

    fmnist = FMNIST(args.bs)
    print('# FMNIST ready', file=open(logname, 'a'), flush=True)

    # models
    j_models = []
    for i in range(n_models):
        j_models += [JModel(nz0[i], nz1[i], nn[i], args.stepsize).to(device)]

    print('# Models prepared', file=open(logname, 'a'), flush=True)

    if args.load_id != '':
        for i in range(10):
            loadprot = load_model(j_models[i], './models/j_' + args.load_id + '_' + str(i) + '.pt', strict=True, load_optimizer=False)
            print('j', i, loadprot, flush=True)
        print('# Models loaded', file=open(logname, 'a'), flush=True)

    log_period = 100
    save_period = 1000
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    loss_c = [0.69] * n_models
    loss_x = [1.4] * n_models

    c_acc_train = torch.ones([n_models], device=device)*0.1
    c_acc_val = torch.ones([n_models], device=device)*0.1

    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor
        
        # learn
        for i in range(n_models): # loop over the models
            # small 
            c, x = fmnist.get_train_w(i, device)
            x = x + torch.randn_like(x)*0.01

            # distillation
            c_dist = []
            x_dist = []
            for j in range(n_models):
                if j == i:
                    continue
                x_dist += [x]
                c_scores = j_models[j].c_model.net(x)
                c_ind = torch.argmax(c_scores, dim=1)
                c_dist += [torch.nn.functional.one_hot(c_ind, num_classes=10).float()]
            c = torch.cat(c_dist + [c])
            x = torch.cat(x_dist + [x])

            lc, lx = j_models[i].optimize(c, x, symmetric=False)
            loss_c[i] = loss_c[i] * afactor1 + lc * afactor
            loss_x[i] = loss_x[i] * afactor1 + lx * afactor

        # once awhile print something out
        if count % log_period == log_period-1:

            # inference
            # new accumulation speed
            count1 = count//log_period
            afactor = (1/(count1 + 1)) if count1 < 10 else 0.1
            afactor1 = 1 - afactor

            for i in range(n_models):
                c_train, x_train = fmnist.get_train_w(i, device)
                dec = inference(x_train, j_models[i])
                gti = torch.argmax(c_train, dim=1)
                deci = torch.argmax(dec, dim=1)
                c_acc_train[i] = c_acc_train[i]*afactor1 + (gti == deci).float().mean()*afactor

                c_val, x_val = fmnist.get_val(1000, device)
                dec = inference(x_val, j_models[i])
                gti = torch.argmax(c_val, dim=1)
                deci = torch.argmax(dec, dim=1)
                c_acc_val[i] = c_acc_val[i]*afactor1 + (gti == deci).float().mean()*afactor

            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            aaa = sum([vlen(j_models[i].c_model).cpu().numpy() for i in range(n_models)])/n_models
            strtoprint += ' clen: ' + str(aaa)

            aaa = sum([loss_c[i].cpu().numpy() for i in range(n_models)])/n_models
            strtoprint += ' closs: ' + str(aaa)

            aaa = sum([c_acc_train[i].cpu().numpy() for i in range(n_models)])/n_models
            strtoprint += ' cacctrain: ' + str(aaa)

            aaa = sum([c_acc_val[i].cpu().numpy() for i in range(n_models)])/n_models
            strtoprint += ' caccval: ' + str(aaa)

            aaa = sum([vlen(j_models[i].x_model).cpu().numpy() for i in range(n_models)])/n_models
            strtoprint += ' xlen: ' + str(aaa)

            aaa = sum([loss_x[i].cpu().numpy() for i in range(n_models)])/n_models
            strtoprint += ' xloss: ' + str(aaa)

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            for i in range(n_models):
                save_model(j_models[i], './models/j_' + args.id + '_' + str(i) + '.pt')

            # image
            c = torch.eye(10, device=device)
            xviz = []
            for i in range(n_models):
                xviz += [j_models[i].x_model.shot(c)]
            xviz = torch.cat(xviz)
            vutils.save_image(xviz, './images/img_' + args.id + '.png', nrow=10)

            # saving c-losses
            with open('./logs/loss_c-' + args.id + '.txt', 'w') as f:
                for i in range(n_models):
                    print(loss_c[i].cpu().numpy(), file=f, flush=True)

            # saving c-accuracies
            with open('./logs/acc_c-' + args.id + '.txt', 'w') as f:
                for i in range(n_models):
                    print(c_acc_train[i].cpu().numpy(), c_acc_val[i].cpu().numpy(), file=f, flush=True)

            # saving x-losses
            with open('./logs/loss_x-' + args.id + '.txt', 'w') as f:
                for i in range(n_models):
                    print(loss_x[i].cpu().numpy(), file=f, flush=True)

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
