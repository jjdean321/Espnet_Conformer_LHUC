#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import math
from torch import nn

import numpy as np


from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding



class Conv2dSubsampling(torch.nn.Module):
    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            #torch.nn.Conv2d(odim, odim, 3, 2),
            #torch.nn.ReLU(),
        )

        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim -1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return x
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model 
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """ x: Input tensor (bacth, time, '*') """
        pe = torch.zeros(x.size(1), self.d_model) # time, dim
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
        )




if __name__ == '__main__':
    
    con2d = Conv2dSubsampling(5, 8, 0.2)
    odim = 8
    x = torch.randn(1, 10, 10)
    z 
    y = con2d.forward(x)
    
    #x = x.unsqueeze(1) 
    print(y.size())
    #x = nn.Conv2d(1, ) 