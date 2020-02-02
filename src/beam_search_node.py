from torch import nn


class BeamSearchNode(nn.Module):
    def __init__(self, hidden_state, previous_node, word_in, logProb, length):
        super(BeamSearchNode, self).__init__()
        """
        @param hidden_state:
        @param previous_node:
        @param word_in:
        @param logProb:
        @param length:
        """
        self.length = length
        self.logProb = logProb
        self.word_in = word_in
        self.previous_node = previous_node
        self.hidden_state = hidden_state

    def forward(self, alpha):
        reward = 0

        return self.logProb / float(self.length - 1 + 1e-6) + alpha * reward
