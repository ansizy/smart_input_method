from torch import nn

import config


class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE, vocab_size)

    def forward(self, x):
        # 1 x shape [batch_size, seq_len]
        embed = self.embedding(x)
        # 2 embed shape [batch_size, seq_len, embedding_dim]
        output, _ = self.rnn(embed)
        # 3 output shape [batch_size, seq_len, hidden_size]
        last_hidden_state = output[:, -1, :]
        # last_hidden_state shape [batch_size, hidden_size]
        res = self.linear(last_hidden_state)
        # res shape [batch_size, vocab_size]

        return res


















