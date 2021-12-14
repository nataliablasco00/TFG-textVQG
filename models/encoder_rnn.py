#Encoder RNN

import torch.nn as nn

from .base_rnn import BaseRNN


class EncoderRNN(BaseRNN):
    

    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0, n_layers=1,
                 bidirectional=False, rnn_cell='lstm', variable_lengths=False):
        
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional,
                                 dropout=dropout_p)
        self.init_weights()

    def init_weights(self):
        
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input_var, input_lengths, h0=None):
        
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        # print("input input_lengths: ---",input_lengths)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                    embedded, input_lengths, batch_first=True,enforce_sorted=False)
        output, hidden = self.rnn(embedded, h0)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                    output, batch_first=True)
        return output, hidden
