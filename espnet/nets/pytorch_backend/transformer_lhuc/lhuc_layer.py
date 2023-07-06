import pdb
import torch
from torch import nn
import math

class LHUC_layer(nn.Module):
    def __init__(self, num_speaker, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_speaker, output_dim)) 
        
    def forward(self, x, spk_id):
        """Add lhuc layer.

        Args:
            x (torch.Tensor): Input tensor (batch, time, output_dim).
            spkid (torch.Tensor): Input tensor (batch, 1).
        Returns:
            linear torch.Tensor: LHUC tensor (batch, time, output_dim).
            spk_id torch.Tensor: speaker tensor (batch, 1).
        """
        spk_id_one_hot = nn.functional.one_hot(spk_id, num_classes=self.weight.shape[0]).squeeze(1).float() # batch, num_speaker
        spk_weight = torch.matmul(spk_id_one_hot, self.weight).unsqueeze(1) # batch, 1, output_dim
        # add 2sigmoid
        spk_weight = 2 * torch.sigmoid(spk_weight)
        linear = x * spk_weight # brocast [batch, tmax, output_dim] if not brocast, could use spk_weight.repeat(1, x.shape[1], 1)
        return linear

