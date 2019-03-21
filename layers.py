import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules import Module
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

__all__ = ['SVDConv2d']


class SVDConv2d(Module):
    '''
    W = UdV
    '''
    def __init__(self, in_channels, out_channels, kernel_size, scale, stride=1,
                padding=0, dilation=1, groups=1, bias=True, norm = False):
        self.eps = 1e-8
        self.norm = norm

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SVDConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.k_scale = scale
        self.total_in_dim = in_channels*kernel_size[0]*kernel_size[1]
        self.weiSize = (self.out_channels,in_channels,kernel_size[0],kernel_size[1])

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = _pair(0)
        self.groups = groups

        self.scale = Parameter(torch.Tensor(1))
        self.scale.data.fill_(1)

        # TODO: set k to min(out,total_in) if not set
        # validation checks on k
        self.k = int(min(self.out_channels, self.total_in_dim)*self.k_scale)
        if self.k == 0:
            self.k = 1
        self.Uweight = Parameter(torch.Tensor(self.out_channels, self.k))#
        self.Dweight = Parameter(torch.Tensor(self.k))#
        self.Vweight = Parameter(torch.Tensor(self.k, self.total_in_dim))#
        self.Uweight.data.normal_(0, math.sqrt(2. / self.out_channels))
        self.Vweight.data.normal_(0, math.sqrt(2. / self.total_in_dim))
        self.Dweight.data.fill_(1)
        self.projectiter = 0
        self.project(style='qr', interval = 1)
        print(self.Uweight.size(),self.Dweight.size(),self.Vweight.size())
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))#
            self.bias.data.fill_(0)
        else:
            self.register_parameter('bias', None)

        if norm:
            self.register_buffer('input_norm_wei',torch.ones(1, in_channels // groups, *kernel_size))

    def update_sigma(self):
        self.Dweight.data = self.Dweight.data/self.Dweight.data.abs().max()

    def spectral_reg(self):
        return -(torch.log(torch.prod(self.Dweight)))

    @property
    def W_(self):
        self.update_sigma()
        return self.Uweight.mm(self.Dweight.diag()).mm(self.Vweight).view(self.weiSize)*self.scale

    def forward(self, input):
        _output = F.conv2d(input, self.W_, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return _output

    def orth_reg(self):
        penalty = 0

        if self.out_channels  <= self.k:
            W = self.Uweight
        else:
            W = self.Uweight.t()
        Wt = torch.t(W)
        WWt = W.mm(Wt)
        I = Variable(torch.eye(WWt.size()[0]).cuda())
        penalty = penalty+((WWt.sub(I))**2).sum()

        W = self.Vweight
        Wt = torch.t(W)
        WWt = W.mm(Wt)
        I = Variable(torch.eye(WWt.size()[0]).cuda())
        penalty = penalty+((WWt.sub(I))**2).sum()
        return penalty

    def project(self, style='none', interval = 1):
        '''
        Project weight to l2 ball
        '''
        self.projectiter = self.projectiter+1
        if style=='qr' and self.projectiter%interval == 0:
            # Compute the qr factorization for U
            if self.out_channels  <= self.k:
                q, r = torch.qr(self.Uweight.data.t())
            else:
                q, r = torch.qr(self.Uweight.data)
            # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
            d = torch.diag(r, 0)
            ph = d.sign()
            q *= ph
            if self.out_channels  <= self.k:
                self.Uweight.data = q.t()
            else:
                self.Uweight.data = q
            
            # Compute the qr factorization for V
            q, r = torch.qr(self.Vweight.data.t())
            # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
            d = torch.diag(r, 0)
            ph = d.sign()
            q *= ph
            self.Vweight.data = q.t()
        elif style=='svd' and self.projectiter%interval == 0:
            # Compute the svd factorization (may be not stable) for U
            u, s, v = torch.svd(self.Uweight.data)
            self.Uweight.data = u.mm(v.t())

            # Compute the svd factorization (may be not stable) for V
            u, s, v = torch.svd(self.Vweight.data)
            self.Vweight.data = u.mm(v.t())

    def showOrthInfo(self):
        s= self.Dweight.data
        _D = self.Dweight.data.diag()
        W = self.Uweight.data.mm(_D).mm(self.Vweight.data)
        _, ss, _ = torch.svd(W.t())
        print('Singular Value Summary: ')
        print('max :',s.max().item(),'max* :',ss.max().item())
        print('mean:',s.mean().item(),'mean*:',ss.mean().item())
        print('min :',s.min().item(),'min* :',ss.min().item())
        print('var :',s.var().item(),'var* :',ss.var().item())
        print('s RMSE: ', ((s-ss)**2).mean().item()**0.5)
        if self.out_channels  <= self.total_in_dim:
            pu = (self.Uweight.data.mm(self.Uweight.data.t())-torch.eye(self.Uweight.size()[0]).cuda()).norm().item()**2
        else:
            pu = (self.Uweight.data.t().mm(self.Uweight.data)-torch.eye(self.Uweight.size()[1]).cuda()).norm().item()**2
        pv =  (self.Vweight.data.mm(self.Vweight.data.t())-torch.eye(self.Vweight.size()[0]).cuda()).norm().item()**2
        print('penalty :', pu, ' (U) + ', pv, ' (V)' )
        return ss
