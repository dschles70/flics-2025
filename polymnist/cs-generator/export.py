import os
import argparse
import torch

from helpers import load_model
from s_model import SModel

parser = argparse.ArgumentParser(description='')
parser.add_argument('--id', required=True, help='Model ID')
parser.add_argument('--nz', required=True, type=int, help='nz')
args = parser.parse_args()

device = torch.cuda.current_device()
torch.autograd.set_detect_anomaly(True)

nc = 10
nz = args.nz

s_model = SModel(nc, nz, 0).to(device)
loadprot = load_model(s_model, './models/s_' + args.id + '.pt')
print('s: ', loadprot, flush=True)

s_model_scripted = torch.jit.script(s_model) # Export to TorchScript
print('... compiled', flush=True)

os.makedirs('./export', exist_ok = True)
s_model_scripted.save('./export/model_' + args.id + '.pt') # Save
print('... saved', flush=True)
