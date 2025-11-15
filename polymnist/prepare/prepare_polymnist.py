import sys
import glob
from PIL import Image
import torch
import torchvision
import pickle

mode = sys.argv[1]

print('Doing for', mode)

images = []
classes = []
styles = []

for d in range(5):
    print('.', end='', flush=True)
    for c in range(10):
        filemask = './data_PM_ICLR_2024/PolyMNIST/' + mode + '/m' + str(d) + '/*.' + str(c) + '.png'
        filenames = glob.glob(filemask)
        print(filemask, len(filenames))
        for filename in filenames:
            with Image.open(filename) as im:
                array = torchvision.transforms.functional.pil_to_tensor(im)/255.
                images += [array]
                classes += [c]
                styles += [d]

images = torch.stack(images)
classes = torch.as_tensor(classes).to(torch.int32)
styles = torch.as_tensor(styles).to(torch.int32)

print()
print('Images:')
print(images.shape)
print(images.dtype)
print(torch.min(images))
print(torch.max(images))

print('Classes:')
print(classes.shape)
print(classes.dtype)
print(torch.min(classes))
print(torch.max(classes))

print('Styles:')
print(styles.shape)
print(styles.dtype)
print(torch.min(styles))
print(torch.max(styles))

dict_to_save = {
    'x' : images.numpy(),
    'c' : classes.numpy(),
    'd' : styles.numpy()}

with open('./polymnist-' + mode + '.pck', 'wb') as f:
    pickle.dump(dict_to_save, f)

print('Done', flush=True)
