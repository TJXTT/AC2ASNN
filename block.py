import torch
import torch.nn as nn
from module import *


class DownSample(nn.Module):

    def __init__(self, H, W, inplanes, planes, T, stride=1, use_bias=False):
        super(DownSample, self).__init__()
        self.T = T
        self.H = H
        self.W = W
        self.conv1x1 = nn.Conv2d(in_channels=inplanes,
                                 out_channels=planes,
                                 kernel_size=1,
                                 stride=stride,
                                 padding=0,
                                 bias=use_bias)
        if stride == 2:
            self.H = int((self.H -1)/stride) + 1
            self.W = int((self.W -1)/stride) + 1

        self.bn = nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.1)

    def forward(self, inp_c, X_s=None, mean=None, var=None, training=False):
        o_c = self.conv1x1(inp_c)
        o_c = self.bn(o_c)
        return o_c

class WDownSample(nn.Module):

    def __init__(self, H, W, inplanes, planes, T, stride=1, use_bias=False):
        super(WDownSample, self).__init__()
        self.T = T
        self.w_conv1x1 = WConv2D(H=H, W=W, in_channels=inplanes, out_channels=planes, kernel_size=1, stride=stride, T=T, padding=0)

    def forward(self, inp_s, weight, bias, steps, mean, var, gamma, beta, training=False):

        mem, o_s, thres, I_s = self.w_conv1x1(inp_s, weight, bias, steps=steps, mean=mean, var=var, gamma=gamma, beta=beta, training=training)

        return I_s, o_s


class VGGBlock_1(nn.Module):
    def __init__(self, H, W, inplanes, planes, kernel, T, stride=1, use_bias=False, downsample=False, pool_type='max', mapping_unit='STSU', kaiming_norm=False):
        super(VGGBlock_1, self).__init__()
        self.padding = kernel // 2
        self.stride = stride
        self.H = H
        self.W = W
        self.T = T
        self.downsample = downsample
        if self.stride == 2:
            
            self.H = int((self.H +2*self.padding-kernel)/self.stride) + 1
            self.W = int((self.W +2*self.padding-kernel)/self.stride) + 1
        self.conv1 = nn.Conv2d(in_channels=inplanes, 
                               out_channels=planes, 
                               kernel_size=kernel, 
                               stride=self.stride, 
                               padding=self.padding, 
                               bias=use_bias)
        self.conv2 = nn.Conv2d(in_channels=planes, 
                               out_channels=planes, 
                               kernel_size=kernel, 
                               stride=self.stride, 
                               padding=self.padding, 
                               bias=use_bias)
        self.bn1 = nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.1)
        self.bn2 = nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.1)
        if pool_type == 'max':
            self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        self.w_conv1 = WConv2D(H=H, W=W, in_channels=inplanes, out_channels=planes, kernel_size=kernel, stride=self.stride, padding=self.padding, T=T)
        self.w_conv2 = WConv2D(H=self.H, W=self.W, in_channels=planes, out_channels=planes, kernel_size=kernel, stride=1, padding=self.padding, T=T, pooling=self.downsample, pool_type=pool_type)
        self.mask1 = 0
        self.mask2 = 0
        
        if mapping_unit == 'STSU':
            self.smp_1 = STSU.apply
            self.smp_2 = STSU.apply
        else:
            self.smp_1 = ReSU.apply
            self.smp_2 = ReSU.apply   

        if kaiming_norm is True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, inp_s, inp_c, steps, epoch, training=False):

        mem, out_s1, thre, I = self.w_conv1(inp_s, self.conv1.weight, self.conv1.bias, steps=steps, mean=self.bn1.running_mean, var=self.bn1.running_var, gamma=self.bn1.weight, beta=self.bn1.bias, training=training)
        out_c = None
        mem_s2, out_s2, thre_s, I_s2 = self.w_conv2(out_s1, self.conv2.weight, self.conv2.bias, steps=steps, mean=self.bn2.running_mean, var=self.bn2.running_var, gamma=self.bn2.weight, beta=self.bn2.bias, training=training)
        if steps == 0:
            self.mask1 = out_s1
            self.mask2 = out_s2
        else:
            self.mask1 = self.mask1 + out_s1
            self.mask2 = self.mask2 + out_s2

        if steps == self.T - 1:
            
            out = self.conv1(inp_c)
            out = self.bn1(out)
            out = nn.functional.dropout(out, p=0.25,training=training)
            out_relu1 = nn.functional.relu(out) 
            out = self.smp_1(out_relu1, self.mask1)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample:
                out = self.pool_layer(out)
            out = nn.functional.dropout(out, p=0.25,training=training)
            out_relu2 = nn.functional.relu(out)
            out_c = self.smp_2(out_relu2, self.mask2)
            
            if training is True:
                Vth_sign = adjust_Vth(out_relu1, self.mask1, self.w_conv1.spike.Vth.data)
                self.w_conv1.spike.Vth.data = 0.9*self.w_conv1.spike.Vth.data + 0.1*Vth_sign
                Vth_sign = adjust_Vth(out_relu2, self.mask2, self.w_conv2.spike.Vth.data)
                self.w_conv2.spike.Vth.data = 0.9*self.w_conv2.spike.Vth.data + 0.1*Vth_sign
            
        return out_s2, out_c


class VGGBlock_2(nn.Module):
    def __init__(self, H, W, inplanes, planes, kernel, T, stride=1, use_bias=False, downsample=False, pool_type='max', mapping_unit='STSU', kaiming_norm=False):
        super(VGGBlock_2, self).__init__()
        self.padding = kernel // 2
        self.stride = stride
        self.H = H
        self.W = W
        self.T = T
        self.downsample = downsample
        self.pool_type = pool_type
        if self.stride == 2:
            
            self.H = int((self.H +2*self.padding-kernel)/self.stride) + 1
            self.W = int((self.W +2*self.padding-kernel)/self.stride) + 1
        self.conv1 = nn.Conv2d(in_channels=inplanes, 
                               out_channels=planes, 
                               kernel_size=kernel, 
                               stride=self.stride, 
                               padding=self.padding, 
                               bias=use_bias)
        self.conv2 = nn.Conv2d(in_channels=planes, 
                               out_channels=planes, 
                               kernel_size=kernel, 
                               stride=self.stride, 
                               padding=self.padding, 
                               bias=use_bias)
        self.conv3 = nn.Conv2d(in_channels=planes, 
                               out_channels=planes, 
                               kernel_size=kernel, 
                               stride=self.stride, 
                               padding=self.padding, 
                               bias=use_bias)
        self.bn1 = nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.1)
        self.bn2 = nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.1)
        self.bn3 = nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.1)
        if pool_type == 'max':
            self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        self.w_conv1 = WConv2D(H=H, W=W, in_channels=inplanes, out_channels=planes, kernel_size=kernel, stride=self.stride, padding=self.padding, T=T)
        self.w_conv2 = WConv2D(H=H, W=W, in_channels=planes, out_channels=planes, kernel_size=kernel, stride=self.stride, padding=self.padding, T=T)
        self.w_conv3 = WConv2D(H=self.H, W=self.W, in_channels=planes, out_channels=planes, kernel_size=kernel, stride=self.stride, padding=self.padding, pooling=self.downsample, pool_type=pool_type, T=T)
        self.mask1 = 0
        self.mask2 = 0
        self.mask3 = 0
        self.mask3_pool = 0

        if mapping_unit == 'STSU':
            self.smp_1 = STSU.apply
            self.smp_2 = STSU.apply
            self.smp_3 = STSU.apply
        else:
            self.smp_1 = ReSU.apply
            self.smp_2 = ReSU.apply
            self.smp_3 = ReSU.apply  

        if kaiming_norm is True:        
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        
    def forward(self, inp_s, inp_c, steps, epoch, training=False):

        mem, out_s1, thre, I = self.w_conv1(inp_s, self.conv1.weight, self.conv1.bias, steps=steps, mean=self.bn1.running_mean, var=self.bn1.running_var, gamma=self.bn1.weight, beta=self.bn1.bias, training=training)
        out_c = None
        mem_s2, out_s2, thre_s, I_s2 = self.w_conv2(out_s1, self.conv2.weight, self.conv2.bias, steps=steps, mean=self.bn2.running_mean, var=self.bn2.running_var, gamma=self.bn2.weight, beta=self.bn2.bias, training=training)
        if self.pool_type  == 'max':
            mem_s3, out_s3, thre_s, I_s3 = self.w_conv3(out_s2, self.conv3.weight, self.conv3.bias, steps=steps, mean=self.bn3.running_mean, var=self.bn3.running_var, gamma=self.bn3.weight, beta=self.bn3.bias, training=training)
        else:
            mem_s3, out_s3, out_s3_ap, thre_s, I_s3 = self.w_conv3(out_s2, self.conv3.weight, self.conv3.bias, steps=steps, mean=self.bn3.running_mean, var=self.bn3.running_var, gamma=self.bn3.weight, beta=self.bn3.bias, training=training)

        if steps == 0:
            self.mask1 = out_s1.data
            self.mask2 = out_s2
            self.mask3 = out_s3
            if self.pool_type != 'max':
                self.mask3_pool = out_s3_ap
        else:
            self.mask1 = self.mask1 + out_s1
            self.mask2 = self.mask2 + out_s2
            self.mask3 = self.mask3 + out_s3
            if self.pool_type != 'max':
                self.mask3_pool = self.mask3_pool + out_s3_ap                

        if steps == self.T - 1:
            
            out = self.conv1(inp_c)
            out = self.bn1(out)
            out_relu1 = nn.functional.relu(out) 
            out = self.smp_1(out_relu1, self.mask1)
            out = self.conv2(out)
            out = self.bn2(out)
            out_relu2 = nn.functional.relu(out)
            out = self.smp_2(out_relu2, self.mask2)

            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample:
                if self.pool_type != 'max':
                    out = nn.functional.relu(out)
                    
                    if training is True:
                        Vth_sign = adjust_Vth(out, self.mask3_pool, self.w_conv3.pool_spike.Vth.data)
                        self.w_conv3.pool_spike.Vth.data = 0.9*self.w_conv3.pool_spike.Vth.data + 0.1*Vth_sign  
                    
                out = self.pool_layer(out)
            out_relu3 = nn.functional.relu(out)
            out_c = self.smp_3(out_relu3, self.mask3)

            if training is True:

                Vth_sign = adjust_Vth(out_relu1, self.mask1, self.w_conv1.spike.Vth.data)
                self.w_conv1.spike.Vth.data = 0.9*self.w_conv1.spike.Vth.data + 0.1*Vth_sign
                Vth_sign = adjust_Vth(out_relu2, self.mask2, self.w_conv2.spike.Vth.data)
                self.w_conv2.spike.Vth.data = 0.9*self.w_conv2.spike.Vth.data + 0.1*Vth_sign
                Vth_sign = adjust_Vth(out_relu3, self.mask3, self.w_conv3.spike.Vth.data)
                self.w_conv3.spike.Vth.data = 0.9*self.w_conv3.spike.Vth.data + 0.1*Vth_sign
            
        return out_s3, out_c


class BasicBlock(nn.Module):

    def __init__(self, H, W, inplanes, planes, kernel, T, stride=1, use_bias=False, downsample=False, mapping_unit='STSU', kaiming_norm=False):
        super(BasicBlock, self).__init__()
        self.padding = kernel // 2
        self.stride = stride
        self.H = H
        self.W = W
        self.T = T
        self.downsample = downsample
        if self.stride == 2:
            self.H = int((self.H +2*self.padding-kernel)/stride) + 1
            self.W = int((self.W +2*self.padding-kernel)/stride) + 1
        self.conv1 = nn.Conv2d(in_channels=inplanes, 
                               out_channels=planes, 
                               kernel_size=kernel, 
                               stride=self.stride, 
                               padding=self.padding, 
                               bias=use_bias)

        self.bn1 = nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, 
                               out_channels=planes, 
                               kernel_size=kernel, 
                               stride=1, 
                               padding=self.padding, 
                               bias=use_bias)

        self.bn2 = nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.1)
        self.w_conv1 = WConv2D(H=H, W=W, in_channels=inplanes, out_channels=planes, kernel_size=kernel, stride=self.stride, padding=self.padding, T=T)
        self.w_conv2 = WConv2D(H=self.H, W=self.W, in_channels=planes, out_channels=planes, kernel_size=kernel, stride=1, padding=self.padding, T=T)
        self.IF = IF(C=planes, H=self.H, W=self.W)
        self.sample = DownSample(H, W, inplanes, planes, T, stride=self.stride, use_bias=use_bias)
        self.wsample = WDownSample(H, W, inplanes, planes, T, stride=self.stride, use_bias=use_bias)
        self.mask1 = 0
        self.mask2 = 0
        self.mask3 = 0

        if mapping_unit == 'STSU':
            self.smp_1 = STSU.apply
            self.smp_2 = STSU.apply
        else:
            self.smp_1 = ReSU.apply
            self.smp_2 = ReSU.apply  

        if kaiming_norm is True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, inp_s, inp_u, inp_c, steps, epoch, training=False):

        with torch.no_grad():
            I_iden_s = inp_s
            if self.downsample is False:
                I_iden_s = inp_u

            out_c = None
            mem, out_s1, thre, I = self.w_conv1(inp_s, self.conv1.weight, self.conv1.bias, steps=steps, mean=self.bn1.running_mean, var=self.bn1.running_var, gamma=self.bn1.weight, beta=self.bn1.bias, training=training)
                
            mem_s2, out_s2, thre_s, I_s2 = self.w_conv2(out_s1, self.conv2.weight, self.conv2.bias, steps=steps, mean=self.bn2.running_mean, var=self.bn2.running_var, gamma=self.bn2.weight, beta=self.bn2.bias, training=training)

            if self.downsample is True:
                I_iden_s, o_s = self.wsample(I_iden_s, self.sample.conv1x1.weight, self.sample.conv1x1.bias, steps=steps, mean=self.sample.bn.running_mean, var=self.sample.bn.running_var, gamma=self.sample.bn.weight, beta=self.sample.bn.bias, training=training)

            out_s = I_s2 + I_iden_s
            out_u, out_s3, _, I_u = self.IF(out_s, steps=steps, training=training)
            #
            if steps == 0:
                self.mask1 = out_s1.data
                self.mask2 = out_s2.data
                self.mask3 = out_s3.data
            else:
                self.mask1 = self.mask1 + out_s1.data
                self.mask2 = self.mask2 + out_s2.data
                self.mask3 = self.mask3 + out_s3.data

        if steps == self.T - 1:

            identity_c = inp_c

            out = self.conv1(inp_c)
            out = self.bn1(out)

            out = nn.functional.dropout(out, p=0.2,training=training)
            out_relu = nn.functional.relu(out) 
            
            out = self.smp_1(out_relu, self.mask1)
            
            if training is True:
                
                Vth_sign = adjust_Vth(out_relu, self.mask1, self.w_conv1.spike.Vth.data)
                self.w_conv1.spike.Vth.data = 0.9*self.w_conv1.spike.Vth.data + 0.1*Vth_sign            

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is True:

                identity_c = self.sample(identity_c, training=training)
                identity_c_relu = nn.functional.relu(identity_c)
                
                if training is True:
                    Vth_sign = adjust_Vth(identity_c, I_iden_s, self.wsample.w_conv1x1.spike.Vth.data)
                    self.wsample.w_conv1x1.spike.Vth.data = 0.9*self.wsample.w_conv1x1.spike.Vth.data + 0.1*Vth_sign
                
            out_c = out + identity_c
            out_c_relu = nn.functional.relu(out_c)
            
            out_c = self.smp_2(out_c_relu, self.mask3)
            
            if training is True:
                Vth_sign = adjust_Vth(out_c_relu, self.mask3, self.IF.Vth.data)
                self.IF.Vth.data = 0.9*self.IF.Vth.data  + 0.1*Vth_sign        
        return out_s3, I_u, out_c

