from block import *
from module import *
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np


class VGG(nn.Module):
    def __init__(self, H=32, W=32, C=3, num_classes=10, 
                 blocks=[VGGBlock_1,VGGBlock_1,VGGBlock_2,
                 VGGBlock_2,VGGBlock_2], 
                 channels=[64,128,256,512,512], 
                 downsample=[False, False, True, True, True], T=5, mapping_unit='STSU', kaiming_norm=False):
        super(VGG, self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.block1 = blocks[0](self.H, self.W, self.C, channels[0], kernel=3, T=T, stride=1, use_bias=True, downsample=downsample[0], mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        if downsample[0]:
            self.H = int((self.H -2)/2) + 1
            self.W = int((self.W -2)/2) + 1
        self.block2 = blocks[1](self.H, self.W, channels[0], channels[1], kernel=3, T=T, stride=1, use_bias=True, downsample=downsample[1], mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        if downsample[1]:
            self.H = int((self.H -2)/2) + 1
            self.W = int((self.W -2)/2) + 1
        self.block3 = blocks[2](self.H, self.W, channels[1], channels[2], kernel=3, T=T, stride=1, use_bias=True, downsample=downsample[2], mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        if downsample[2]:
            self.H = int((self.H -2)/2) + 1
            self.W = int((self.W -2)/2) + 1
        self.block4 = blocks[3](self.H, self.W, channels[2], channels[3], kernel=3, T=T, stride=1, use_bias=True, downsample=downsample[3], mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        if downsample[3]:
            self.H = int((self.H -2)/2) + 1
            self.W = int((self.W -2)/2) + 1
        self.block5 = blocks[4](self.H, self.W, channels[3], channels[4], kernel=3, T=T, stride=1, use_bias=True, downsample=downsample[4], mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        if downsample[4]:
            self.H = int((self.H -2)/2) + 1
            self.W = int((self.W -2)/2) + 1

        self.cls = nn.Linear(self.H*self.W*channels[4], num_classes)
        self.num_classes = num_classes

    def forward(self, input, steps=100, epoch=100, training=True):
        with torch.no_grad():
            spike_logits = Variable(torch.zeros(input.size(0), 512*self.H*self.W).cuda())
        I_0 = None
        for i in range(steps):
            out_s, out_c = self.block1(input, input, i, epoch=epoch, training=training)
            out_s, out_c = self.block2(out_s, out_c, i, epoch=epoch, training=training)
            out_s, out_c = self.block3(out_s, out_c, i, epoch=epoch, training=training)
            out_s, out_c = self.block4(out_s, out_c, i, epoch=epoch, training=training)
            out_s, out_c = self.block5(out_s, out_c, i, epoch=epoch, training=training)
            out_s = out_s.view(out_s.size(0),-1)
            if i == steps - 1:
                out_c = out_c.view(out_c.size(0),-1)

            if i == 0:
                spike_logits = out_s
            else:
                spike_logits = spike_logits + out_s
        logit_sp = self.cls(spike_logits/steps)
        logit_sp_single = self.cls(out_s)
        logit = self.cls(out_c/steps)
        return logit, logit_sp


class ResNet(nn.Module):
    def __init__(self, H=32, W=32, C=3, num_classes=10, strides=[1,1,1,2,2], channels=[64,64,128,256,512], T=5, mapping_unit='STSU', kaiming_norm=False):
        super(ResNet, self).__init__()

        padding=1
        kernel_size=3
        downsample = False
        self.epoch = 1
        self.steps = T# torch.nn.Parameter(torch.ones(1).cuda(), requires_grad=True)
        self.ori_H = H
        self.ori_W = W
        self.ch0 = channels[0]
        self.H = H
        self.W = W
        self.num_classes = num_classes

        self.cnn00 = nn.Conv2d(in_channels=C, out_channels=channels[0], kernel_size=3, stride=strides[0], padding=1, bias=True) # encoder
        self.w_cnn00 = WConv2D(H=self.H, W=self.W, in_channels=C, out_channels=channels[0], kernel_size=3, stride=strides[0], padding=1) # encoder
        self.H = int((self.H +2*padding-kernel_size)/strides[0]) + 1
        self.W = int((self.W +2*padding-kernel_size)/strides[0]) + 1
        if channels[0] != channels[1] or strides[1]==2:
            downsample = True
        self.bn00 = nn.BatchNorm2d(channels[0], affine=True, track_running_stats=True, momentum=0.1) # Instance_Normalize(64)
        self.block11 = BasicBlock(self.H, self.W, channels[1], channels[1], 3, T=T, stride=strides[1], use_bias=True, downsample=downsample, mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        self.H = int((self.H +2*padding-kernel_size)/strides[1]) + 1
        self.W = int((self.W +2*padding-kernel_size)/strides[1]) + 1
        self.block12 = BasicBlock(self.H, self.W, channels[1], channels[1], 3, T=T, stride=1, use_bias=True, downsample=False, mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        self.block21 = BasicBlock(self.H, self.W, channels[1], channels[2], 3, T=T, stride=strides[2], use_bias=True, downsample=True, mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        self.H = int((self.H +2*padding-kernel_size)/strides[2]) + 1
        self.W = int((self.W +2*padding-kernel_size)/strides[2]) + 1
        self.block22 = BasicBlock(self.H, self.W, channels[2], channels[2], 3, T=T, stride=1, use_bias=True, downsample=False, mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        self.block31 = BasicBlock(self.H, self.W, channels[2], channels[3], 3, T=T, stride=strides[3], use_bias=True, downsample=True, mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        self.H = int((self.H +2*padding-kernel_size)/strides[3]) + 1
        self.W = int((self.W +2*padding-kernel_size)/strides[3]) + 1
        self.block32 = BasicBlock(self.H, self.W, channels[3], channels[3], 3, T=T, stride=1, use_bias=True, downsample=False, mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        self.block41 = BasicBlock(self.H, self.W, channels[3], channels[4], 3, T=T, stride=strides[4], use_bias=True, downsample=True, mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        self.H = int((self.H +2*padding-kernel_size)/strides[4]) + 1
        self.W = int((self.W +2*padding-kernel_size)/strides[4]) + 1
        self.block42 = BasicBlock(self.H, self.W, channels[4], channels[4], 3, T=T, stride=1, use_bias=True, downsample=False, mapping_unit=mapping_unit, kaiming_norm=kaiming_norm)
        self.fc2 = nn.Linear(self.H*self.W*channels[4], num_classes, bias=True)
        self.mask0 = 0
        
        if mapping_unit == 'STSU':
            self.smp = STSU.apply
        else:
            self.smp = ReSU.apply

        
        if kaiming_norm is True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
               
    def forward(self, input, steps=100, epoch=100, training=True):
        self.epoch = epoch + 1
        with torch.no_grad():
            spike_logits = Variable(torch.zeros(input.size(0), self.num_classes).cuda())
        I_0 = None

        for i in range(steps):
            cnt1 = 0
            mem_0, out_0, thre_0, I_0s = self.w_cnn00(input, self.cnn00.weight, self.cnn00.bias, steps=i, mean=self.bn00.running_mean, var=self.bn00.running_var, gamma=self.bn00.weight, beta=self.bn00.bias, training=training)
            if i == 0:
                self.mask0 = out_0
            else:
                self.mask0 = self.mask0 + out_0
            if i == steps - 1:
                I_0 = self.cnn00(input)
                I_0 = self.bn00(I_0)
                I_0 = nn.functional.dropout(I_0,p=0.2,training=training)
                I_0_relu = nn.functional.relu(I_0) 

                I_0 = self.smp(I_0_relu, self.mask0)
                
                if training is True:
                    Vth_sign = adjust_Vth(I_0_relu, self.mask0,self.w_cnn00.spike.Vth.data)
                    self.w_cnn00.spike.Vth.data = 0.9*self.w_cnn00.spike.Vth.data + 0.1*Vth_sign
                

            out_s, out_u, out_c = self.block11(out_0, I_0s, I_0, i, epoch=epoch, training=training)
            out_s, out_u, out_c = self.block12(out_s, out_u, out_c, i, epoch=epoch, training=training)
            out_s, out_u, out_c = self.block21(out_s, out_u, out_c, i, epoch=epoch, training=training)
            out_s, out_u, out_c = self.block22(out_s, out_u, out_c, i, epoch=epoch, training=training)
            out_s, out_u, out_c = self.block31(out_s, out_u, out_c, i, epoch=epoch, training=training)
            out_s, out_u, out_c = self.block32(out_s, out_u, out_c, i, epoch=epoch, training=training)
            out_s, out_u, out_c = self.block41(out_s, out_u, out_c, i, epoch=epoch, training=training)
            out_s, out_u, out_c = self.block42(out_s, out_u, out_c, i, epoch=epoch, training=training)
            out = out_s.view(out_s.size(0),-1)

            last_emb = self.fc2(out)
            spike_logits = spike_logits + last_emb
        

        out_c = out_c.view(out_c.size(0), -1)

        output_cnn = self.fc2(out_c/steps)

        output = spike_logits/steps


        return output_cnn, output.data
