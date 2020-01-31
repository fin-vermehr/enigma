import torch
import torch.nn.functional as F
from torch import nn


class Decoder(nn.Module):
    def __init__(
            self,
            embedding_size,
            hidden_state_size,
            output_size,
            number_of_layers,
            dropout):

        super(Decoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_state_size
        self.dropout = dropout
        self.number_of_layers = number_of_layers
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(embedding_size, hidden_state_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_state_size,
                          hidden_state_size,
                          number_of_layers,
                          dropout=dropout)
        self.concatenation_layer = nn.Linear(hidden_state_size * number_of_layers,
                                             hidden_state_size)
        self.output_layer = nn.Linear(hidden_state_size, output_size)

    def forward(self, previous_step, last_hidden_state, encoder_outputs):
        # Get embeddings
        embedded = self.embedding(previous_step)
        # add dropout
        embedded = self.dropout(embedded)
        predicted_output, gru_hidden_state = self.gru(embedded, last_hidden_state)
        # Attention Weights
        attention_weights = self.attention(predicted_output, encoder_outputs)
        # Get context vector
        # BMM performs a batch matrix-matrix product of matrices
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        context_vector = attention_weights.bmm(encoder_outputs.transpose(0, 1))
        predicted_and_context = torch.cat((
            predicted_output.squeeze(0), context_vector.squeeze(1)), dim=1
        )

        # Tanh concatenation layer output then predict next character
        output = self.output_layer(
            torch.tanh(self.concatenation_layer(predicted_and_context))
        )
        return F.softmax(output, dim=1), gru_hidden_state

    @staticmethod
    def attention(hidden, encoder_outputs):
        attn_energies = torch.sum(hidden * encoder_outputs, dim=2).t()
        # Softmax scores to probability
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
