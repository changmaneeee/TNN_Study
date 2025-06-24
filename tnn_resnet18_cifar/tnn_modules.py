#TNN Modules

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ternarize(tensor):
    """
    Ternarize the input tensor to -1, 0, 1 based on the threshold.
    If |W_ij| < delta_i, then W_ij_t = 0
    If W_ij >= delta_i, then W_ij_t = +alpha_i
    If W_ij <= -delta_i, then W_ij_t = -alpha_i
    """
   output = torch.zeros_linke(tensor)
   delta = 0.7*tensor.avs().mean() # Threshold for ternarization

   output[tensor > delta] = 1
   output[tensor < -delta] = -1

   return output


class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input_tensor):
        return ternarize(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient is passed through as is, since we are using STE
        return grad_output


class TernaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(TernaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight_fp = nn.Parameter(self.weight.data.clone())
    
    def forward(self, x):
        ternary_weight = TernaryQuantizeSTE.apply(self.weight_fp)

        output = F.conv2d(x, ternary_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def reset_parameters(self):
        super().resnet_parameters()
        if hasattr(self, 'weight_fp'):
            self.weight_fp.data = self.weight.data.clone()

class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(TernaryLinear, self).__init__(in_features, out_features, bias)
        self.weight_fp = nn.Parameter(self.weight.data.clone())

    def forward(self,x):
        ternary_weight = TernaryQuantizeSTE.apply(self.weight_fp)
        output = F.linear(x, ternary_weight, self.bias)
        return output
        

