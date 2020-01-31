from typing import Optional, Tuple

import torch.nn as nn
from dynaconf import settings
from torch import Tensor


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

        # Bidirectional GRU!
        self.gru = nn.GRU(hidden_size, hidden_size, number_of_layers, dropout=dropout, bidirectional=True)

    def forward(self,
                sequence: Tensor,
                input_lengths: Tensor,
                hidden: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        # Convert word indexes to embeddings
        # |Sequence|: SeqLen x Batch
        embeddings = self.embedding(sequence)

        # enforce_sorted would sort our embeddings from largest to smallest. Don't need that.
        pad_packed = nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, enforce_sorted=False)

        outputs, hidden = self.gru(pad_packed, hidden)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum each direction together
        outputs = outputs[:, :, :outputs.shape[2] // 2] + outputs[:, :, outputs.shape[2] // 2:]
        # Return output and final hidden state
        return outputs, hidden
