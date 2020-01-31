import torch


class MaskNLLLoss:
    def __init__(self, input_sequence, target_sequence, mask):
        self.loss = -torch.log(torch.gather(input_sequence, 1, target_sequence.view(-1, 1)).squeeze(1))

    def loss(self):