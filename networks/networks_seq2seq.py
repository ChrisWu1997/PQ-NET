import torch
import torch.nn as nn
import random


# RNN
##############################################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, n_layer, bidirectional=bidirectional, dropout=0.2 if n_layer==2 else 0)

        self.init_hidden = self.initHidden()

    def forward(self, input, init_hidden):
        """

        :param input: (seq_len, batch_size, feature_dim)
        :return:
            output: (seq_len, batch, num_directions * hidden_size)
            h_n: (num_layers * num_directions, batch, hidden_size)
        """
        output, hidden = self.gru(input, init_hidden)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(self.n_layer * self.num_directions, batch_size, self.hidden_size, requires_grad=False)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.n_units_hidden1 = 256
        self.n_units_hidden2 = 128

        self.gru = nn.GRU(input_size, hidden_size, n_layer, bidirectional=bidirectional, dropout=0.2 if n_layer==2 else 0)
        self.linear1 = nn.Sequential(nn.Linear(hidden_size, self.n_units_hidden1),
                                     nn.LeakyReLU(True),
                                     nn.Linear(self.n_units_hidden1, input_size - 6),
                                     # nn.Sigmoid()
                                     )
        self.linear2 = nn.Sequential(nn.Linear(hidden_size, self.n_units_hidden2),
                                     nn.ReLU(True),
                                     nn.Dropout(0.2),
                                     nn.Linear(self.n_units_hidden2, 6),
                                     # nn.Sigmoid()
                                     )
        self.linear3 = nn.Sequential(nn.Linear(hidden_size, self.n_units_hidden2),
                                     nn.ReLU(True),
                                     nn.Dropout(0.2),
                                     nn.Linear(self.n_units_hidden2, 1),
                                     # nn.Sigmoid()
                                     )

        self.lockdrop = LockedDropout()
        self.dropout_i = 0.2
        self.dropout_o = 0.2

        self.init_input = self.initInput()

    def forward(self, input, hidden):
        """

        :param input: (1, batch, output_size)
        :param hidden: initial hidden state
        :return:
            output: (1, batch, num_directions * hidden_size)
            hidden: (num_layers * 1, batch, hidden_size)
            output_seq: (batch, 1 * output_size)
            stop_sign: (batch, 1)
        """
        # seq_len, batch_size = input.size(0), input.size(1)
        input = self.lockdrop(input, self.dropout_i)
        output, hidden = self.gru(input, hidden)
        # hidden : (num_layers * 1, batch, hidden_size)
        hidden1, hidden2 = torch.split(hidden, 1, 0)
        # output_ = self.lockdrop(output, self.dropout_o)
        output_code = self.linear1(hidden1.squeeze(0))
        output_param = self.linear2(hidden2.squeeze(0))
        stop_sign = self.linear3(hidden1.squeeze(0))
        output_seq = torch.cat([output_code, output_param], dim=1)

        return output, hidden, output_seq, stop_sign

    def initInput(self):
        initial_code = torch.zeros((1, 1, self.input_size - 6), requires_grad=False)
        initial_param = torch.tensor([0.5, 0.5, 0.5, 1, 1, 1], dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)
        initial = torch.cat([initial_code, initial_param], dim=2)
        return initial


# Seq2Seq
##############################################################################
class Seq2SeqAE(nn.Module):
    def __init__(self, en_input_size, de_input_size, hidden_size):
        super(Seq2SeqAE, self).__init__()
        self.n_layer = 2
        self.encoder = EncoderRNN(en_input_size, hidden_size, n_layer=self.n_layer, bidirectional=True)
        self.decoder = DecoderRNN(de_input_size, hidden_size * 2, n_layer=self.n_layer, bidirectional=False)
        self.max_length = 10

    def infer_encoder(self, input_seq, batch_size=1):
        """
        :param input_seq: (n_parts, 1, feature_dim)
        :return:
            h_n: (num_layers * num_directions, batch, hidden_size)
        """
        encoder_init_hidden = self.encoder.init_hidden.repeat(1, batch_size, 1).cuda()
        _, hidden = self.encoder(input_seq, encoder_init_hidden)
        hidden = hidden.view(self.n_layer, 2, batch_size, -1)
        hidden0, hidden1 = torch.split(hidden, 1, 1)
        hidden = torch.cat([hidden0.squeeze(1), hidden1.squeeze(1)], 2)
        # hidden = hidden[-1]
        return hidden

    def infer_decoder(self, decoder_hidden, target_seq, teacher_forcing_ratio=0.5):
        batch_size = target_seq.size(1)
        target_length = target_seq.size(0)
        decoder_input = self.decoder.init_input.detach().repeat(1, batch_size, 1).cuda()

        # Teacher forcing: Feed the target as the next input
        # Without teacher forcing: use its own predictions as the next input
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        stop_signs = []
        for di in range(target_length):
            decoder_output, decoder_hidden, output_seq, stop_sign = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(output_seq)
            stop_signs.append(stop_sign)
            # using target seq as input or not
            decoder_input = target_seq[di:di+1] if use_teacher_forcing else output_seq.detach().unsqueeze(0)
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        stop_signs = torch.stack(stop_signs, dim=0)
        return decoder_outputs, stop_signs

    def infer_decoder_stop(self, decoder_hidden, length=None):
        decoder_outputs = []
        stop_signs = []
        decoder_input = self.decoder.init_input.detach().repeat(1, 1, 1).cuda()
        for di in range(self.max_length):
            decoder_output, decoder_hidden, output_seq, stop_sign = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(output_seq)
            stop_signs.append(stop_sign)
            if length is not None:
                if di == length - 1:
                    break
            elif torch.sigmoid(stop_sign[0, 0]) > 0.5:
                # stop condition
                break
            decoder_input = output_seq.detach().unsqueeze(0)  # using target seq as input
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        stop_signs = torch.stack(stop_signs, dim=0)
        return decoder_outputs, stop_signs

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        """

        :param input_seq: (seq_len, batch_size, feature_dim) PackedSequence
        :param target_seq: (seq_len, batch_size, feature_dim)
        :param teacher_forcing_ratio: float
        :return:
            decoder_outputs: (seq_len, batch, num_directions, output_size)
            stop_signs: (seq_len, batch, num_directions, 1)
        """
        batch_size = target_seq.size(1)
        encoder_hidden = self.infer_encoder(input_seq, batch_size)
        decoder_hidden = encoder_hidden # .unsqueeze(0)
        decoder_outputs, stop_signs = self.infer_decoder(decoder_hidden, target_seq, teacher_forcing_ratio)
        return decoder_outputs, stop_signs


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        # mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = m.detach().clone().requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


if __name__ == '__main__':
    pass
