from typing import Tuple

import torch
from torch import nn


class MaskedNLLLoss(nn.Module):
    """
    Loss on a masked input
    """

    def __init__(self):
        super(MaskedNLLLoss, self).__init__()

    def forward(self,
                input_tensor: torch.Tensor,
                target_tensor: torch.Tensor,
                mask: torch.Tensor
                ) -> Tuple[torch.Tensor, int]:
        """
        One forward iteration over a single step in a batch

        @param input_tensor: (batch_size x embedding_size)
        @param target_tensor: (batch_size x 1)
        @param mask: (batch_size)
        @return:
        """

        total_items = mask.sum().item()
        cross_entropy = -torch.log(
            torch.gather(input_tensor, 1, target_tensor.view(-1, 1)).squeeze(1)
        )
        loss = cross_entropy.masked_select(mask).mean()

        return loss, total_items
