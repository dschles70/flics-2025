import os
import torch
import torchvision.utils as vutils

from mnist import MNIST

# entry point, the main
if __name__ == '__main__':

    # printout
    print('# Start ...', flush=True)

    bs = 1000 # batch size

    mnist = MNIST()
    print('# Dataset loaded.', flush=True)

    # prepare folders
    os.makedirs('./generated_data/origin1/first/', exist_ok=True)
    os.makedirs('./generated_data/origin1/second/', exist_ok=True)

    print('0 to 4')

    xall = [mnist.x[i] for i in range(5)]
    xall = torch.cat(xall)
    xall = torch.split(xall, bs) # data batches

    for b in range(len(xall)):
        # current batch
        xcurr = xall[b]

        # save original images
        for i in range(xcurr.shape[0]):
            vutils.save_image(xcurr[i], './generated_data/origin1/first/img0_%05d.png' % (b*bs+i))

        print('.', flush=True, end='')

    print()
    
    print('5 to 9')

    xall = [mnist.x[i + 5] for i in range(5)]
    xall = torch.cat(xall)
    xall = torch.split(xall, bs) # data batches

    for b in range(len(xall)):
        # current batch
        xcurr = xall[b]

        # save original images
        for i in range(xcurr.shape[0]):
            vutils.save_image(xcurr[i], './generated_data/origin1/second/img1_%05d.png' % (b*bs+i))

        print('.', flush=True, end='')

    print(' done.')
