import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from scipy.stats import ortho_group
from utils.options import args


class _Quantize(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, step,th):
        ctx.step= step.item()
        ctx.th = th.item()         
        output = input.clone().zero_()
        
        output[input.ge(ctx.th)] = 1
        output[input.le(ctx.th)] = -1
        
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step
        
        return grad_input, None,None
                
quantize1 = _Quantize.apply

class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        #self.N_bits = 7
        #step = self.weight.abs().max()/((2**self.N_bits-1))
        #self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
        #self.weight.data = quantize(self.weight, self.step).data.clone() 
        #self.biase=torch.Tensor(torch.ones(self.weight.size())).fill_(0).cuda()
        #self.mask=nn.Parameter(torch.Tensor(torch.ones(self.weight.size())).fill_(1).cuda())
        
    
        
        
    def forward(self, input):
       
        self.N_bits = 1
        th= self.weight.mean()
        
        step = self.weight[self.weight.ge(th)+self.weight.le(-th)].abs().mean()
        
        QW = quantize1(self.weight, step,th)
        
        return F.linear(input, QW*step, self.bias)


class BinarizeConv2d(nn.Conv2d):
    
    def __init__(self,*kargs,no_train =0,**kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.no_train = no_train
        self.k = torch.tensor([10.]).float()
        self.t = torch.tensor([0.1]).float()
        self.epoch = -1
        #self.mask=nn.Parameter(torch.Tensor((torch.randn(self.weight.size()[0])+1)/100000),requires_grad=True)
        #if no_train == 0:
        #    self.mask[int(self.weight.size()[0]/2):].data[:] = 
        #print(self.mask.sum())
        
        w = self.weight
        #self.quan =  nn.Parameter(BinaryQuantize_m().apply(self.mask, self.k.to(w.device), self.t.to(w.device))[:,None,None,None].detach(),requires_grad=False)
        
        
        self.a, self.b = get_ab(np.prod(w.shape[1:]))
        R1 = torch.tensor(ortho_group.rvs(dim=self.a)).float().cuda()
        R2 = torch.tensor(ortho_group.rvs(dim=self.b)).float().cuda()
        self.register_buffer('R1', R1)
        self.register_buffer('R2', R2)
        self.Rweight = torch.ones_like(w)
        self.masking_enable = nn.Parameter(torch.ones(1), requires_grad=False)
        sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1).detach()
        self.alpha = nn.Parameter(sw.cuda(), requires_grad=True)
        self.rotate = nn.Parameter(torch.ones(w.size(0), 1, 1, 1).cuda()*np.pi/2, requires_grad=True)
        self.Rotate = torch.zeros(1)
    

    def forward(self, input,gg= 0):
        a0 = input
        w = self.weight
        w1 = w - w.mean([1,2,3], keepdim=True)
        w2 = w1 / w1.std([1,2,3], keepdim=True)
        a1 = a0 - a0.mean([1,2,3], keepdim=True)
        a2 = a1 / a1.std([1,2,3], keepdim=True)
        a, b = self.a, self.b
        X = w2.view(w.shape[0], a, b)
        if self.epoch > -1 and self.epoch % args.rotation_update == 0:
            for _ in range(3):
                #* update B
                V = self.R1.t() @ X.detach() @ self.R2
                B = torch.sign(V)
                #* update R1
                D1 = sum([Bi@(self.R2.t())@(Xi.t()) for (Bi,Xi) in zip(B,X.detach())])
                U1, S1, V1 = torch.svd(D1)
                self.R1 = (V1@(U1.t()))
                #* update R2
                D2 = sum([(Xi.t())@self.R1@Bi for (Xi,Bi) in zip(X.detach(),B)])
                U2, S2, V2 = torch.svd(D2)
                self.R2 = (U2@(V2.t()))
        self.Rweight = ((self.R1.t())@X@(self.R2)).view_as(w)
        delta = self.Rweight.detach() - w2
        w3 = w2 + torch.abs(torch.sin(self.rotate)) * delta
        
        #* binarize
        bw = BinaryQuantize().apply(w3, self.k.to(w.device), self.t.to(w.device))
        #print(self.mask)
        '''if self.masking_enable == 1:
            masking = BinaryQuantize_m().apply(self.mask, self.k.to(w.device), self.t.to(w.device))
            with torch.no_grad():
                self.quan.data = masking[:,None,None,None].clone().detach()
            bw = bw * masking[:,None,None,None]'''
            
        
       
        
        if input.size()[1]==3:
            ba = a2
        else:
            ba = BinaryQuantize_a().apply(a2, self.k.to(w.device), self.t.to(w.device))
            #ba = a2
        #* 1bit conv
        
        #print(ba[:,0:(c1-12),:,:].size())
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
            
        #* scaling factor
        output = output * self.alpha
        return output
        
class BinaryQuantize_m2(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (torch.sign(input)+1)/2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        
        grad_input =  grad_output.clone()/2
        return grad_input/2

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = torch.tensor(1.).to(input.device)
        t = max(t, torch.tensor(1.).to(input.device))
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None

class BinaryQuantize_m(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = (torch.sign(input)+1)/2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = torch.tensor(1.).to(input.device)
        t = max(t, torch.tensor(1.).to(input.device))
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input/2, None, None


def get_ab(N):
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i
