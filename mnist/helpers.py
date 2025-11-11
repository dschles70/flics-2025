import os
import torch

def vlen(layer):
    vsum = 0.
    vnn = 0
    for vv in layer.parameters():
        if vv.requires_grad:
            param = vv.data
            vsum = vsum + (param*param).sum()
            vnn = vnn + param.numel()
    return vsum/vnn

def save_model(model : torch.nn.Module, filename: str):
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict()
        }

        torch.save(checkpoint, filename)

def load_model(model : torch.nn.Module, filename : str, strict : bool = True, load_optimizer : bool = True) -> str:
        
        checkpoint = torch.load(filename)
        loadprot_m = model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if load_optimizer:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return loadprot_m
