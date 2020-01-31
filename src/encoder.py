from typing import Optional, Tuple

from dynaconf import settings
from torch import Tensor
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            hidden_size: int,
            number_of_layers: int = 1,
            dropout=0,
    ):
        super(Encoder, self).__init__()
        self.number_of_layers = number_of_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(embedding_size, hidden_size, padding_idx=settings.PADDING_INDEX)

        self.gru = nn.GRU(hidden_size, hidden_size, number_of_layers, dropout=dropout, bidirectional=True)

    def forward(self, sequence: Tensor, input_lengths: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Convert word indexes to embeddings
        # |Sequence|: SeqLen x Batch
        embeddings = self.embedding(sequence).cuda()

        pad_packed = nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, enforce_sorted=False)

        outputs, hidden = self.gru(pad_packed, hidden)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
