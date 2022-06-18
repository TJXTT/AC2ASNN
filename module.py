import torch
import torch.nn as nn
import numpy as np

def adjust_Vth(x_in, x_sp, Vth, scale=0.1, tor=0.1):

    x_in_relu = nn.functional.relu(x_in)
    ann_pos = x_in_relu.gt(0).type(torch.cuda.FloatTensor)
    ann_neg_pos = 1 - ann_pos
    snn_pos = x_sp.gt(0).type(torch.cuda.FloatTensor)
    err_act_pos = ann_neg_pos * snn_pos

    err_act_pos = torch.mean(err_act_pos).gt(tor).type(torch.cuda.FloatTensor)
    thre = Vth + scale*err_act_pos

    return thre

class STSU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_in, x_sp):

        x_out_f = x_sp.data

        return x_out_f

    @staticmethod
    def backward(ctx, g):
        grad = g
        return grad, None, None


class ReSU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_in, x_sp):

        x_out_f = x_sp.data * x_in.gt(0).type(torch.cuda.FloatTensor)

        return x_out_f

    @staticmethod
    def backward(ctx, g):

        grad = g#*scale

        return grad, None, None

class IF(nn.Module):


    def __init__(self, C, H, W, dp_rate=0.75):
        super(IF, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.mem = torch.zeros((1)).cuda()
        self.steps = 0
        self.Vth = nn.Parameter(torch.rand(1,self.C, self.H, self.W), requires_grad=False)

    def reset_parameters(self, nums):
        self.mem = torch.zeros((1)).cuda()

    def forward(self, input, steps, training=True):
        self.steps = steps

        out = self.IF_Neuron(self.mem, input.data, training=training)

        return self.mem, out, self.Vth, input.data


    def IF_Neuron(self, membrane_potential, I, training=True):
        # check exceed membrane potential and reset
        if self.steps == 0:
            mp_output = I
        else:
            mp_output = self.mem.data + I 
        out = (mp_output - self.Vth).gt(0).type(torch.cuda.FloatTensor)


        m_o = mp_output*(1-out)

        self.mem = nn.functional.dropout(m_o,p=0.5,training=training) 
        return out


class WConv2D(nn.Module):


    def __init__(self, H, W, in_channels, out_channels, kernel_size, stride, padding, T=5, pooling=False, pool_type='max'):
        super(WConv2D, self).__init__()
        self.H = H
        self.W = W
        self.T = T
        self.pooling = pooling
        self.C_in = in_channels
        self.C_out = out_channels
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        
        if stride > 1:
            self.H = int((self.H +2*padding-kernel_size)/stride) + 1
            self.W = int((self.W +2*padding-kernel_size)/stride) + 1
        
        if self.pooling:
            self.pool_spike = IF(C=self.C_out, H=self.H, W=self.W)
            self.H = self.H // 2
            self.W = self.W // 2
            
            if pool_type == 'max':
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
            else:
            	self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.spike = IF(C=self.C_out, H=self.H, W=self.W)
        self.pool_type = pool_type
        self.w_pos = 0
        self.w_neg = 0
        self.b = 0
        

    def forward(self, inp, weight=None, bias=None, groups=None, pooling=False, steps=0, mean=None, var=None, gamma=None, beta=None, training=True):
        with torch.no_grad():
            a = gamma.data/torch.sqrt(var.data+1e-5)

            self.w_pos = (weight.data)* a.view(-1,1,1,1)
            self.b = a*(bias-mean) + beta.data
            self.b = self.b / self.T
            inp = inp.data
            if groups is None:
                out_pos = nn.functional.conv2d(input=inp, weight=self.w_pos, bias=self.b, stride=self.stride, padding=self.padding)
            else:
                out_pos = nn.functional.conv2d(input=inp, weight=self.w_pos, bias=self.b, groups=groups, stride=self.stride, padding=self.padding)
            out = out_pos.data
            
            if self.pooling is True:
                if self.pool_type == 'max':
                	out = self.pool(out)
                else:
                	mem, out_pool, thre, I = self.pool_spike(out, steps=steps, training=training)
                	out = self.pool(out_pool)
                	mem, out, thre, I = self.spike(out, steps=steps, training=training)
                	return mem, out, out_pool, thre, I
            mem, out, thre, I = self.spike(out, steps=steps, training=training)
            return mem, out, thre, I
