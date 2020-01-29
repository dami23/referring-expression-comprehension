import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
import numpy as np

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional = False,
                 input_dropout = 0, lstm_dropout = 0, n_layers = 1, rnn_type = 'lstm'):
        super(BiLSTMEncoder, self).__init__()

        self.num_glimpse = 2
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear_trans = nn.Sequential(nn.Linear(word_vec_size * 2, word_embedding_size),
                                          nn.Tanh())
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size), nn.ReLU())
        self.rnn_type = rnn_type
        self.lstm = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers, batch_first=True,
                                                  bidirectional=bidirectional, dropout=lstm_dropout)

    def forward(self, input_labels):
        input_lengths = (input_labels != 0).sum(1)
        input_lengths_list = input_lengths.data.cpu().numpy().tolist()
        sorted_input_lengths = np.sort(input_lengths_list)[::-1].tolist()

        sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()
        s2r = {s: r for r, s in enumerate(sort_ixs)}
        recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]

        sort_ixs = input_labels.data.new(sort_ixs).long()
        recover_ixs = input_labels.data.new(recover_ixs).long()
        input_sents = input_labels[sort_ixs]

        embedded = self.embedding(input_sents)
        embedded = self.input_dropout(embedded)
        embedded = self.mlp(embedded)

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths, batch_first=True)
        output, hidden = self.lstm(embedded)

        embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
        embedded = embedded[recover_ixs]

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[recover_ixs]

        hidden = hidden[0]
        hidden = hidden[:, recover_ixs, :]
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(hidden.size(0), -1)

        return output, hidden, embedded

class PhraseAttention(nn.Module):
    def __init__(self, input_dim):
        super(PhraseAttention, self).__init__()

        self.fc = nn.Linear(input_dim, 1)

    def forward(self, output, embed, input_labels):
        cxt_score = self.fc(output).squeeze(2)
        atten = F.softmax(cxt_score, dim = 1)

        is_not_zero = (input_labels != 0).float()

        atten = atten * is_not_zero
        atten = atten / atten.sum(1).view(atten.size(0), 1).expand(atten.size(0), atten.size(1))

        attn3 = atten.unsqueeze(1)
        weighted_emb = torch.bmm(attn3, embed)
        weighted_emb = weighted_emb.squeeze(1)

        return atten, weighted_emb
