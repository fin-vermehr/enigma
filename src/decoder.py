import torch
from torch import nn
from torch.nn.functional import relu, log_softmax, softmax


class Decoder(nn.Module):
    #TODO Remvoe hard code max_length
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=42):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=2)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = relu(output)
        output, hidden = self.gru(output, hidden)

        output = log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    # TODO: Fix device and initialize random
    def initialize_hidden_state(self):
        return torch.rand(2, 1, self.hidden_size,).cuda()
