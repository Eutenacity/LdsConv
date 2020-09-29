import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torch.nn import init
class Learned_Dw_Conv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, fiter_kernel, stride, padding, dropout_rate, k,cardinality=8):
        super(Learned_Dw_Conv, self).__init__()

        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.group=self.out_channels//self.cardinality
        self.in_channel_per_group=in_channels//self.group
        self.dwconv = nn.Conv2d(in_channels, out_channels, fiter_kernel, stride, padding, bias=False,groups=self.group)

        self.delta_prune = (int)(self.in_channel_per_group*(self.cardinality-k)*0.25)

        self.tmp = self.group*(self.in_channel_per_group*self.cardinality- 4*self.delta_prune)
        self.pwconv = nn.Conv2d(self.tmp, out_channels, 1, 1, bias=False)

        self.dwconv2 = nn.Conv2d(self.tmp, self.tmp, fiter_kernel, stride, padding, groups=self.tmp, bias=False)
        self.register_buffer('index', torch.LongTensor(self.tmp))
        self.register_buffer('_mask_dw', torch.ones(self.dwconv.weight.size()))
        self.register_buffer('_count', torch.zeros(1))
        #self.pwconv.weight.requires_grad = False
        #self.dwconv2.weight.requires_grad = False
    def _check_drop(self):
        progress = Learned_Dw_Conv.global_progress
        if progress == 0:
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress<100 :
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress>100 :
            self.dwconv.weight.data.zero_()
            ### Check for dropping
        if progress == 25 or progress == 50 or progress ==75  or progress == 100: # or progress==150 :
            # if progress == 1 or progress == 2 or progress == 3 or progress == 4:
            if progress<=100:
                self._dropping_group(self.delta_prune)
         #   else:
         #       if self.in_channel_per_group==8:
        #            self._dropping_group(16)
         #       else:
        #           self._dropping_group(32)
        return
    def _dropping_group(self,delta):
        if Learned_Dw_Conv.global_progress <= 100:
            weight=self.dwconv.weight*self.mask_dw
            weight=weight.view(self.group,self.cardinality,self.in_channel_per_group,3,3).abs().sum([3,4])
            for i in range(self.group):
                weight_tmp=weight[i,:,:].view(-1)
                di=weight_tmp.sort()[1][self.count:self.count+delta]
                for d in di.data:
                    out_ = d // self.in_channel_per_group
                    in_ = d % self.in_channel_per_group
                    self._mask_dw[i*self.cardinality+out_, in_, :, :].fill_(0)
            self.count = self.count + delta
            #print(self.in_channel_per_group)
            #print(self.delta_prune)
            #print(self.count)
        index=0
        if Learned_Dw_Conv.global_progress == 100:
            print_mask=self.mask_dw[:,:,0,0].view(self.group,self.cardinality,self.in_channel_per_group).permute(0,2,1).sum(-1)
            line='%s\n'%print_mask
            with open('testlogs.txt','a')as f:
                f.write(line)
            self.pwconv.weight.data.zero_()
            for i in range(self.group):
                for j in range(self.cardinality):
                    for k in range(self.in_channel_per_group):
                        if self._mask_dw[i*self.cardinality+j,k,0,0]==1:
                            self.index[index]=i*self.in_channel_per_group+k
                            self.dwconv2.weight.data[index,:,:,:]=self.dwconv.weight.data[i*self.cardinality+j,k,:,:].view(1,3,3)
                            self.pwconv.weight.data[i*self.cardinality+j,index,:,:].fill_(1)
                            index=index+1
            assert index==self.tmp
            self.dwconv.weight.data.zero_()
    def forward(self, x):
        progress = Learned_Dw_Conv.global_progress
        self._check_drop()
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output

        if progress < 100:
            weight = self.dwconv.weight * self.mask_dw
            return F.conv2d(x,weight, None, self.dwconv.stride,
                            1, self.dwconv.dilation, self.group)
        else:
            x = torch.index_select(x, 1, Variable(self.index))
            x = self.dwconv2(x)
            self.pwconv.weight.data = self.pwconv.weight.data  # *self.mask_pw
            x = F.conv2d(x, self.pwconv.weight, None, self.pwconv.stride,
                         0, self.pwconv.dilation, 1)
            return x

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def mask_dw(self):
        return Variable(self._mask_dw)

    @property
    def mask_pw(self):
        return Variable(self._mask_pw)
    @property
    def ldw_loss(self):
        if Learned_Dw_Conv.global_progress >= 100:
            return 0
        weight = self.dwconv.weight * self.mask_dw
        weight_1=weight.abs().sum(-1).sum(-1).view(self.group,self.cardinality,self.in_channel_per_group)
        weight=weight.abs().sum([2,3]).view(self.group,-1)
        mask=torch.ge(weight,torch.topk(weight,2*self.in_channel_per_group,1,sorted=True)[0][:,2*self.in_channel_per_group-1]
                      .view(self.group,1).expand_as(weight)).view(self.group,self.cardinality,self.in_channel_per_group)\
            .sum(1).view(self.group,1,self.in_channel_per_group)
        mask = torch.exp((mask.float() - 1.5 * self.k) / (3)) - 1
        mask=mask.expand_as(weight_1)
        weight=(weight_1.pow(2)*mask).sum(1).clamp(min=1e-6).sum(-1).sum(-1).sqrt()
        return weight
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        def conv_ldw(inp,oup,stride):
            return nn.Sequential(

                Learned_Dw_Conv(inp,oup//4,3,stride,1,0,k=2),
                nn.BatchNorm2d(oup//4),
                nn.ReLU(inplace=True),

                nn.Conv2d(oup//4, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw_new(inp, oup, stride):
            return  nn.Sequential(
                nn.Conv2d(inp, inp*2, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp*2),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp*2, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_ldw(32, 64, 1),
            conv_ldw(64, 128, 2),
            conv_ldw(128, 128, 1),
            conv_ldw(128, 256, 2),
            conv_ldw(256, 256, 1),
            conv_ldw(256, 512, 2),
            conv_ldw(512, 512, 1),
            conv_ldw(512, 512, 1),
            conv_ldw(512, 512, 1),
            conv_ldw(512, 512, 1),
            conv_ldw(512, 512, 1),
            conv_ldw(512, 1024, 2),
            conv_ldw(1024, 1024, 1),
            nn.AvgPool2d(8),
        )
        self.fc = nn.Linear(1024, 1000)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x,progress=None):
        if progress != None:
            Learned_Dw_Conv.global_progress=progress
        else:
            Learned_Dw_Conv.global_progress =600.1
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
