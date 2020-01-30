import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            hidden_size: int,
    ):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(embedding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, sequence, hidden_state):
        embeddings = self.embedding(sequence).view(1, 1, -1)
        output = embeddings
        output, hidden = self.gru(output, hidden_state)
        return output, hidden

    def initialize_hidden_state(self):
        #TODO: initialize to something else and remove cuda?
        return torch.rand(1, 1, self.hidden_size).cuda()
