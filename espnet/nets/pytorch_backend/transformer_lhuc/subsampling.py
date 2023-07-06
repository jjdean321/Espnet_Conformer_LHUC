#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""
import pdb
import torch

from espnet.nets.pytorch_backend.transformer_lhuc.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer_lhuc.lhuc_layer import LHUC_layer

class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message (str): Message for error catch
        actual_size (int): the short size that cannot pass the subsampling
        limit (int): the limit size for subsampling

    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv2dSubsampling2) and size < 3:
        return True, 3
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, lhuc_layers, speaker_num, lhuc_dim, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.lhuc_layers = lhuc_layers
        self.lhuc_dim = lhuc_dim
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )

        if "lhuc-conv2d-1" in self.lhuc_layers:
            self.lhuc_layer_conv1 = LHUC_layer(speaker_num, odim * ((idim - 1) // 2))
        if "lhuc-conv2d-2" in self.lhuc_layers:
            self.lhuc_layer_conv2 = LHUC_layer(speaker_num, odim * (((idim - 1) // 2 - 1) // 2)) # lhuc_layer after conv2d
        if "lhuc-conv2d-linear" in self.lhuc_layers:
            self.lhuc_layer_linear = LHUC_layer(speaker_num, odim) 
        if "lhuc-conv2d-PE" in self.lhuc_layers:
            self.lhuc_layer_pe = LHUC_layer(speaker_num, odim) 

        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.out = pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)

    def forward(self, x, x_mask, spk_id):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            spk_id (torch.Tensor): Speaker id (#batch, 1).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv1(x)
        b, c, t, f = x.size() 
        if "lhuc-conv2d-1" in self.lhuc_layers:
            x = self.lhuc_layer_conv1(x.transpose(1, 2).contiguous().view(b, t, c * f), spk_id) # b, t, c*f
            x = x.view(b, t, c, f).transpose(1, 2).contiguous()

        x = self.conv2(x)

        b, c, t, f = x.size()       
        #
        if "lhuc-conv2d-2" in self.lhuc_layers:
            x = self.lhuc_layer_conv2(x.transpose(1, 2).contiguous().view(b, t, c * f), spk_id)
            x = self.linear(x)
        else:
            x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if "lhuc-conv2d-linear" in self.lhuc_layers:
            x = self.lhuc_layer_linear(x, spk_id)
            # print("conv2d-linear")

        x = self.out(x)

        if "lhuc-conv2d-PE" in self.lhuc_layers:
            x = self.lhuc_layer_pe(x, spk_id)        

        if x_mask is None:
            return x, None

        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
