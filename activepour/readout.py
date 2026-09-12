"""Matched MLP heads, a plain CNN baseline, and parameter-free feature pooling."""
import torch
from torch import nn
from torch.nn import functional as F

class Head(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(1664,128),nn.SiLU(),nn.Dropout(.15),
                         nn.Linear(128,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1),nn.Sigmoid())

class PlainEncoder(nn.Module):
    def __init__(self):
        super().__init__();layers=[];before=5
        for after in [32,64,96,96]:
            layers.extend([nn.Conv2d(before,after,3,stride=2,padding=1),nn.GroupNorm(8,after),nn.SiLU()]);before=after
        self.cnn=nn.Sequential(*layers)
        self.action=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,128))
        self.register_buffer('action_low',torch.tensor([45.,.6,.1]))
        self.register_buffer('action_high',torch.tensor([100.,2.,1.5]))
    def forward(self,probe,action):
        state=F.avg_pool2d(self.cnn(probe.float()),4,4).flatten(1)
        a=2*(action-self.action_low)/(self.action_high-self.action_low)-1
        return torch.cat([state,self.action(a)],1)

class Direct(nn.Module):
    def __init__(self,head=None):
        super().__init__();self.head=head if head is not None else Head();self.encoder=PlainEncoder()
    def forward(self,probe,action):return self.head(self.encoder(probe,action))

def pooled_future(z):
    if z.shape[1:]!=(16,64,64):raise ValueError('Expected final SD3.5 latent')
    b=len(z)
    q=torch.stack([z[:,:,:32,:32],z[:,:,:32,32:],z[:,:,32:,:32],z[:,:,32:,32:]],1)
    return F.avg_pool2d(q.reshape(b*4,16,32,32),8,8).reshape(b,1024)

def frozen_features(conditioner,probe,action,z):
    if conditioner.training:raise ValueError('Feature extraction requires eval mode')
    h,_,_=conditioner.temporal_features(probe.float(),action.float())
    a=conditioner.action_encoder[:2](conditioner.normalize_action(action.float()))
    return torch.cat([h.mean((-2,-1)).flatten(1),a,pooled_future(z.float())],1)

def normalize(x,stats,zero_future=False):
    y=(x-stats['mean'].to(x.device))/stats['std'].to(x.device)
    if zero_future:y=torch.cat([y[:,:640],torch.zeros_like(y[:,640:])],1)
    return y
