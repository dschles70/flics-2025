import os
import argparse
import time
import torch

from helpers import vlen, load_model, save_model

from c_model import CModel

from polymnist import POLYMNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', required=True, help='')
    parser.add_argument('--load_id', default='', help='')
    parser.add_argument('--stepsize', type=float, default=1e-4, help='')
    parser.add_argument('--bs', type=int, default=256, help='')
    parser.add_argument('--niterations', type=int, default=-1, help='')

    args = parser.parse_args()

    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    time0 = time.time()
    print(os.uname(), flush=True)

    # printout
    logname = './logs/log-' + args.id + '.txt'
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    nc = 10

    polymnist = POLYMNIST(args.bs)
    print('# POLYMNIST ready', file=open(logname, 'a'), flush=True)

    # models
    c_model = CModel(nc, args.stepsize).to(device)
    
    print('# Models prepared', file=open(logname, 'a'), flush=True)

    if args.load_id != '':
        loadprot = load_model(c_model, './models/c_' + args.load_id + '.pt', load_optimizer=False)
        print('c: ', loadprot, flush=True)
        
        print('# Models loaded', file=open(logname, 'a'), flush=True)


    log_period = 100
    save_period = 1000
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    c_loss = torch.ones([], device=device)

    c_acc_train = torch.ones([], device=device) / nc
    c_acc_val = torch.ones([], device=device) / nc
    
    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        c, _, x = polymnist.get_train(device)

        c_loss = c_loss * afactor1 + c_model.optimize(c, x) * afactor

        # once awhile print something out
        if count % log_period == log_period-1:

            # inference
            # accumulation speed
            count1 = count//log_period
            afactor = (1/(count1 + 1)) if count1 < 10 else 0.1
            afactor1 = 1 - afactor

            c_inf, _, x_inf = polymnist.get_train(device)
            c_inf = torch.argmax(c_inf, dim=1)
            with torch.no_grad():
                c_scores = c_model.net(x_inf)
                c_dec = torch.argmax(c_scores, dim=1)
            c_acc_train = c_acc_train*afactor1 + (c_inf==c_dec).float().mean()*afactor

            c_inf, _, x_inf = polymnist.get_val(device)
            c_inf = torch.argmax(c_inf, dim=1)
            with torch.no_grad():
                c_scores = c_model.net(x_inf)
                c_dec = torch.argmax(c_scores, dim=1)
            c_acc_val = c_acc_val*afactor1 + (c_inf==c_dec).float().mean()*afactor

            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' clen: ' + str(vlen(c_model).cpu().numpy())
            strtoprint += ' closs: ' + str(c_loss.cpu().numpy())
            strtoprint += ' cacctrain: ' + str(c_acc_train.cpu().numpy())
            strtoprint += ' caccval: ' + str(c_acc_val.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            save_model(c_model, './models/c_' + args.id + '.pt')

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
