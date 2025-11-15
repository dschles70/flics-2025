import os
import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import vlen, load_model, save_model

from c_model import CModel
from d_model import DModel
from z_model import ZModel
from s_model import SModel
from x_model import XModel

from polymnist import POLYMNIST
from cs_source import CSSOURCE

def complete_cdx(
        c : torch.Tensor,
        d : torch.Tensor,
        x : torch.Tensor,
        z_model : ZModel,
        s_model : SModel):
    
    bs = c.shape[0]
    z = z_model.sample_random(bs)

    for _ in range(10):
        s = s_model.sample(c, d, z, x)
        z = z_model.sample(c, d, s, x)

    c0, _ = torch.chunk(c, 2)
    d0, _ = torch.chunk(d, 2)
    z0, _ = torch.chunk(z, 2)
    s0, s1 = torch.chunk(s, 2)
    x0, _ = torch.chunk(x, 2)

    s0 = s_model.sample(c0, d0, z0, x0)
    l0 = torch.ones([s0.shape[0]], device=s0.device).long()
    l1 = torch.zeros([s1.shape[0]], device=s1.device).long()

    s = torch.cat((s0, s1))
    l = torch.cat((l0, l1))

    return z, s, l

def complete_cds(
        c : torch.Tensor,
        d : torch.Tensor,
        s : torch.Tensor,
        z_model : ZModel,
        x_model : XModel):
    
    bs = c.shape[0]
    z = z_model.sample_random(bs)

    for _ in range(10):
        x = x_model.sample(c, d, z, s)
        z = z_model.sample(c, d, s, x)

    c0, _ = torch.chunk(c, 2)
    d0, _ = torch.chunk(d, 2)
    z0, _ = torch.chunk(z, 2)
    s0, _ = torch.chunk(s, 2)
    x0, x1 = torch.chunk(x, 2)

    x0 = x_model.sample(c0, d0, z0, s0)
    l0 = (torch.ones([x0.shape[0]], device=x0.device) * 2).long()
    l1 = torch.zeros([x1.shape[0]], device=x1.device).long()

    x = torch.cat((x0, x1))
    l = torch.cat((l0, l1))

    return z, x, l

def inference(
        x : torch.Tensor,
        c_model : CModel,
        d_model : DModel,
        z_model : ZModel,
        s_model : SModel):
    
    with torch.no_grad():
        bs = x.shape[0]
        c = c_model.sample_random(bs)
        d = d_model.sample_random(bs)
        z = z_model.sample_random(bs)
        s = s_model.sample_random(bs)

        cmarg = torch.ones_like(c) / c_model.nc
        dmarg = torch.ones_like(d) / d_model.nd

        for _ in range(100):
            for op in torch.randperm(4):
                if op == 0:
                    c_scores = c_model.net(d, z, s, x)
                    c_probs = torch.softmax(c_scores, dim=1)
                    cmarg = cmarg * 0.9 + c_probs * 0.1
                    c = c_model.sample(d, z, s, x)
                elif op == 1:
                    d_scores = d_model.net(c, z, s, x)
                    d_probs = torch.softmax(d_scores, dim=1)
                    dmarg = dmarg * 0.9 + d_probs * 0.1
                    d = d_model.sample(c, z, s, x)
                elif op == 2:
                    z = z_model.sample(c, d, s, x)
                else:
                    s = s_model.sample(c, d, z, x)
                
        cdec = torch.argmax(cmarg, dim=1)
        ddec = torch.argmax(dmarg, dim=1)

        return cdec, ddec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', required=True, help='')
    parser.add_argument('--load_id', default='', help='')
    parser.add_argument('--stepsize', type=float, default=1e-4, help='Gradient step size.')
    parser.add_argument('--bs', type=int, default=256, help='Batch size')
    parser.add_argument('--niterations', type=int, default=-1, help='')
    parser.add_argument('--ny', type=int, required=True, help='')
    parser.add_argument('--nz', type=int, required=True, help='')
    parser.add_argument('--generator_path', default='', help='')

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
    nd = 5
    ny = args.ny
    nz = args.nz

    cs_source = CSSOURCE(args.bs, args.generator_path, device)
    print('# CSSource ready', file=open(logname, 'a'), flush=True)

    polymnist = POLYMNIST(args.bs)
    print('# POLYMNIST ready', file=open(logname, 'a'), flush=True)

    means = polymnist.means.to(device)
    std = polymnist.std.to(device)
    iconst = (1 - 2 * means) / std # this is used for inversion of the foreground pixels

    # models
    c_model = CModel(nc, nd, nz, args.stepsize).to(device)
    d_model = DModel(nc, nd, nz, args.stepsize, iconst).to(device)
    z_model = ZModel(nc, nd, nz, args.stepsize).to(device)
    s_model = SModel(nc, nd, nz, args.stepsize).to(device)
    x_model = XModel(nc, nd, ny, nz, args.stepsize, iconst).to(device)
    
    print('# Models prepared', file=open(logname, 'a'), flush=True)

    if args.load_id != '':
        loadprot = load_model(c_model, './models/c_' + args.load_id + '.pt', load_optimizer=False)
        print('c: ', loadprot, flush=True)
        loadprot = load_model(d_model, './models/d_' + args.load_id + '.pt', load_optimizer=False)
        print('d: ', loadprot, flush=True)
        loadprot = load_model(z_model, './models/z_' + args.load_id + '.pt', load_optimizer=False)
        print('z: ', loadprot, flush=True)
        loadprot = load_model(s_model, './models/s_' + args.load_id + '.pt', load_optimizer=False)
        print('s: ', loadprot, flush=True)
        loadprot = load_model(x_model, './models/x_' + args.load_id + '.pt', load_optimizer=False)
        print('x: ', loadprot, flush=True)
        
        print('# Models loaded', file=open(logname, 'a'), flush=True)

    log_period = 100
    save_period = 1000
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    c_loss = torch.ones([], device=device)
    d_loss = torch.ones([], device=device)
    z_loss = torch.ones([], device=device)
    s_loss = torch.ones([], device=device)
    x_loss = torch.ones([], device=device)

    c_acc_train = torch.ones([], device=device) / nc
    d_acc_train = torch.ones([], device=device) / nd
    c_acc_val = torch.ones([], device=device) / nc
    d_acc_val = torch.ones([], device=device) / nd

    c_sta = c_model.sample_random(12)
    d_sta = d_model.sample_random(12)
    z_sta = z_model.sample_random(12)
    s_sta = s_model.sample_random(12)
    x_sta = x_model.sample(c_sta, d_sta, z_sta, s_sta)
    
    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        # get and complete data
        c0, d0, x0 = polymnist.get_train(device)
        z0, s0, l0 = complete_cdx(c0, d0, x0, z_model, s_model)

        c1, s1 = cs_source.get_train(device)
        d1 = d_model.sample_random(args.bs)
        z1, x1, l1 = complete_cds(c1, d1, s1, z_model, x_model)

        c = torch.cat((c0, c1))
        d = torch.cat((d0, d1))
        z = torch.cat((z0, z1))
        s = torch.cat((s0, s1))
        x = torch.cat((x0, x1))
        l = torch.cat((l0, l1))

        c_loss = c_loss * afactor1 + c_model.optimize(c, d, z, s, x) * afactor
        d_loss = d_loss * afactor1 + d_model.optimize(c, d, z, s, x) * afactor
        z_loss = z_loss * afactor1 + z_model.optimize(c, d, z, s, x, l) * afactor
        s_loss = s_loss * afactor1 + s_model.optimize(c, d, z, s, x, l) * afactor
        x_loss = x_loss * afactor1 + x_model.optimize(c, d, z, s, x, l) * afactor

        # update limiting
        for op in torch.randperm(5):
            if op == 0:
                c_sta = c_model.sample(d_sta, z_sta, s_sta, x_sta)
            elif op == 1:
                d_sta = d_model.sample(c_sta, z_sta, s_sta, x_sta)
            elif op ==2:
                z_sta = z_model.sample(c_sta, d_sta, s_sta, x_sta)
            elif op == 3:
                s_sta = s_model.sample(c_sta, d_sta, z_sta, x_sta)
            else:
                x_sta = x_model.sample(c_sta, d_sta, z_sta, s_sta)

        # once awhile print something out
        if count % log_period == log_period-1:

            # inference
            # accumulation speed
            count1 = count//log_period
            afactor = (1/(count1 + 1)) if count1 < 10 else 0.1
            afactor1 = 1 - afactor

            c_inf, d_inf, x_inf = polymnist.get_train(device)
            c_inf = torch.argmax(c_inf, dim=1)
            d_inf = torch.argmax(d_inf, dim=1)
            c_dec, d_dec = inference(x_inf, c_model, d_model, z_model, s_model)
            c_acc_train = c_acc_train*afactor1 + (c_inf==c_dec).float().mean()*afactor
            d_acc_train = d_acc_train*afactor1 + (d_inf==d_dec).float().mean()*afactor

            c_inf, d_inf, x_inf = polymnist.get_val(device)
            c_inf = torch.argmax(c_inf, dim=1)
            d_inf = torch.argmax(d_inf, dim=1)
            c_dec, d_dec = inference(x_inf, c_model, d_model, z_model, s_model)
            c_acc_val = c_acc_val*afactor1 + (c_inf==c_dec).float().mean()*afactor
            d_acc_val = d_acc_val*afactor1 + (d_inf==d_dec).float().mean()*afactor

            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' clen: ' + str(vlen(c_model).cpu().numpy())
            strtoprint += ' closs: ' + str(c_loss.cpu().numpy())
            strtoprint += ' cacctrain: ' + str(c_acc_train.cpu().numpy())
            strtoprint += ' caccval: ' + str(c_acc_val.cpu().numpy())

            strtoprint += ' dlen: ' + str(vlen(d_model).cpu().numpy())
            strtoprint += ' dloss: ' + str(d_loss.cpu().numpy())
            strtoprint += ' dacctrain: ' + str(d_acc_train.cpu().numpy())
            strtoprint += ' daccval: ' + str(d_acc_val.cpu().numpy())

            strtoprint += ' zlen: ' + str(vlen(z_model).cpu().numpy())
            strtoprint += ' zloss: ' + str(z_loss.cpu().numpy())

            strtoprint += ' slen: ' + str(vlen(s_model).cpu().numpy())
            strtoprint += ' sloss: ' + str(s_loss.cpu().numpy())

            strtoprint += ' xlen: ' + str(vlen(x_model).cpu().numpy())
            strtoprint += ' xloss: ' + str(x_loss.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            save_model(c_model, './models/c_' + args.id + '.pt')
            save_model(d_model, './models/d_' + args.id + '.pt')
            save_model(z_model, './models/z_' + args.id + '.pt')
            save_model(s_model, './models/s_' + args.id + '.pt')
            save_model(x_model, './models/x_' + args.id + '.pt')

            # image
            im1 = x0[0:12] * std + means
            im2 = s_model.get_dec(c0[0:12], d0[0:12], z0[0:12], x0[0:12]).expand_as(im1)
            im3 = x_model.reconstruct(c0[0:12], d0[0:12], z0[0:12], s0[0:12], x[0:12]) * std + means
            im4 = (s1[0:12]).expand_as(im1)
            im5 = x_model.get_ms(c1[0:12], d1[0:12], z1[0:12], s1[0:12])[0] * std + means
            im6 = s_model.get_dec(c_sta, d_sta, z_sta, x_sta).expand_as(im1)
            im7 = x_model.get_ms(c_sta, d_sta, z_sta, s_sta)[0] * std + means

            xviz = torch.cat((im1, im2, im3, im4, im5, im6, im7))
            vutils.save_image(xviz, './images/img_' + args.id + '.png', nrow=12)

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
