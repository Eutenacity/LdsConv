import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
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

        self.delta_prune = (int)(self.in_channel_per_group*(self.cardinality-k)*0.125)

        self.tmp = self.group*(self.in_channel_per_group*self.cardinality- 8*self.delta_prune)
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
        if progress<300 :
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress>300 :
            self.dwconv.weight.data.zero_()
            ### Check for dropping
        if progress == 37 or progress == 75 or progress == 112 or progress == 150 or progress == 187 or progress == 225 or progress == 262 or progress == 300: 
            
            self._dropping_group(self.delta_prune)
        
        return
    def _dropping_group(self,delta):
        if Learned_Dw_Conv.global_progress <= 300:
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
        if Learned_Dw_Conv.global_progress == 300:
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

        if progress < 300:
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
        return 0
        if Learned_Dw_Conv.global_progress >= 300:
            return 0
        weight = self.dwconv.weight * self.mask_dw
        weight_1=weight.abs().sum(-1).sum(-1).view(self.group,self.cardinality,self.in_channel_per_group)
        weight=weight.abs().sum([2,3]).view(self.group,-1)
        mask=torch.ge(weight,torch.topk(weight,self.k*self.in_channel_per_group,1,sorted=True)[0][:,self.k*self.in_channel_per_group-1]
                      .view(self.group,1).expand_as(weight)).view(self.group,self.cardinality,self.in_channel_per_group)\
            .sum(1).view(self.group,1,self.in_channel_per_group)
        mask = torch.exp((mask.float() - 1.5 * self.k) / (10)) - 1
        mask=mask.expand_as(weight_1)
        weight=(weight_1.pow(2)*mask).sum(1).clamp(min=1e-6).sum(-1).sum(-1)
        return weight

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2=Learned_Dw_Conv(inter_planes,out_planes,3,1,1,0,2,cardinality=12)
        #self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth=100, num_classes=100, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x,progress=300.1):
        Learned_Dw_Conv.global_progress=progress
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
