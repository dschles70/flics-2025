import argparse
import numpy as np
import torch
import torchvision.utils as vutils

from hvae import HVAE

from helpers import ensure_dir

# entry point, the main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--call_prefix', required=True, help='Call prefix.')
    parser.add_argument('--nz0', type=int, required=True, help='')
    parser.add_argument('--nz1', type=int, required=True, help='')

    args = parser.parse_args()

    # printout
    print('# Start ...', flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    bs = 1000 # batch size
    nbatches = 60
    nz0 = args.nz0
    nz1 = args.nz1
    call_prefix = args.call_prefix

    # the model
    model = HVAE(nz0, nz1, 0).to(device)

    loadprot = model.load_state_dict(torch.load('./models/' + call_prefix + '.pt'), strict=True)
    print('# Model: ', loadprot, flush=True)

    # prepare folders
    ensure_dir('./generated_data/' + call_prefix + '/')
    
    for b in range(nbatches):

        # single-shot images
        x = (model.single_shot(bs)>0.5).float()
        for i in range(bs):
            vutils.save_image(x[i], './generated_data/' + call_prefix + '/img_%05d.png' % (b*bs+i))

        print('.', flush=True, end='')

    print(' done.')
