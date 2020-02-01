import torch
import torch.nn.functional as F
from torch import nn


class Decoder(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            hidden_state_size: int,
            output_size: int,
            number_of_layers: int,
            dropout: float):
        """
        Uni-directional, batched, attention decoder that iterates over the sequence_length of a batched tensor

        @param embedding_size:
        @param hidden_state_size:
        @param output_size:
        @param number_of_layers:
        @param dropout: probability of dropout
        """
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
        self.concatenation_layer = nn.Linear(hidden_state_size * 2,
                                             hidden_state_size)
        self.output_layer = nn.Linear(hidden_state_size, output_size)

    def forward(self, previous_step, last_hidden_state, encoder_outputs):
        """
        Feedworward one batch tensor

        @param previous_step: (1 x batch_size)
        @param last_hidden_state: (encoder_num_layers x batch_size x hidden_size)
        @param encoder_outputs: (embedding_size x batch_size x hidden_size)
        @return:
        """

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
        cat = self.concatenation_layer(predicted_and_context)
        output = self.output_layer(torch.tanh(cat))
        return F.softmax(output, dim=1), gru_hidden_state

    @staticmethod
    def attention(hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        @param hidden: (num_decoder_layers x batch_size x hidden_size)
        @param encoder_outputs: (embedding_size x batch_size x hidden_size)
        @return scores: (batch_size x 1 x embedding_size)
        """
        attention_score = torch.sum(hidden * encoder_outputs, dim=2).t()
        # Softmax scores to probability
        return F.softmax(attention_score, dim=1).unsqueeze(1)
