#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch
from espnet.nets.pytorch_backend.transformer_lhuc.lhuc_layer import LHUC_layer


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, lhuc_layers="", speaker_num=1, lnum=-1, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

        # lhuc 
        self.lhuc_layers = lhuc_layers

        self.lhuc_layer_name_ff_1 = "lhuc-enc-" + str(lnum) + "-" + "ff-1"
        self.lhuc_layer_name_ff_2 = "lhuc-enc-" + str(lnum) + "-" + "ff-2"

        if self.lhuc_layer_name_ff_1 in self.lhuc_layers:
            self.lhuc_layer_ff_1 = LHUC_layer(speaker_num, hidden_units)
        
        if self.lhuc_layer_name_ff_2 in self.lhuc_layers:
            self.lhuc_layer_ff_2 = LHUC_layer(speaker_num, idim)

    def forward(self, x, spk_id=-1):
        """Forward function."""
        x = self.activation(self.w_1(x))

        if self.lhuc_layer_name_ff_1 in self.lhuc_layers:
            x = self.lhuc_layer_ff_1(x, spk_id)
        
        x = self.dropout(x)
        x = self.w_2(x)

        if self.lhuc_layer_name_ff_2 in self.lhuc_layers:
            x = self.lhuc_layer_ff_2(x, spk_id)

        return x
