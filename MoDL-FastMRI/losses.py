import torch

def nrmse(x,y):
    num = torch.norm(x-y,p=2)
    den = torch.norm(x,p=2)
    return num/den

def nrmae(x,y):
    num = torch.norm(x-y,p=1)
    den = torch.norm(x,p=1)
    return num/den

def mmse(x,y):
    num = torch.norm(x-y)
    return num