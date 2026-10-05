
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
from torch.nn.utils.rnn import pack_padded_sequence

__all__ = [ 'BERTEncoder', 'ROBERTAEncoder']

class BERTEncoder(nn.Module):

    def __init__(self, args):

        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(args.text_pretrained_model)

    def forward(self, text_feats):
        outputs = self.bert(text_feats[:, 0], text_feats[:, 1], text_feats[:, 2])
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

class RoBERTaEncoder(nn.Module):

    def __init__(self, args):

        super(RoBERTaEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(args.text_pretrained_model)

    def forward(self, text_feats):
        outputs = self.roberta(text_feats[:, 0], text_feats[:, 1], text_feats[:, 2])
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

class SubNet(nn.Module):
    """Feed-forward subnetwork for pre-fusion video and audio features."""

    def __init__(self, in_size, hidden_size, dropout):
        """Initialize the feature projection layers."""
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """Project one batch of features."""
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        """Initialize an LSTM feature encoder."""
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        """Encode padded sequences using their true lengths."""
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze(0))
        y_1 = self.linear_1(h)
        return y_1
