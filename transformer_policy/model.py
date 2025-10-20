import math
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Multi-head attention implementation
# Based on: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries
        self.in_decoder = in_decoder

        # Linear projections for Q, K, V
        self.cast_queries = nn.Linear(d_model, n_heads * d_queries)
        self.cast_keys_values = nn.Linear(d_model, n_heads * (d_queries + d_values))
        self.cast_output = nn.Linear(n_heads * d_values, d_model)

        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, query_sequences, key_value_sequences, key_value_sequence_lengths):
        batch_size = query_sequences.size(0)
        q_len = query_sequences.size(1)
        kv_len = key_value_sequences.size(1)

        self_attention = torch.equal(key_value_sequences, query_sequences)
        input_to_add = query_sequences.clone()

        # layer norm
        query_sequences = self.layer_norm(query_sequences)
        if self_attention:
            key_value_sequences = self.layer_norm(key_value_sequences)

        # Get Q, K, V
        queries = self.cast_queries(query_sequences)
        keys, values = self.cast_keys_values(key_value_sequences).split(
            split_size=self.n_heads * self.d_keys, dim=-1
        )

        # Reshape for multi-head attention
        queries = queries.contiguous().view(batch_size, q_len, self.n_heads, self.d_queries)
        keys = keys.contiguous().view(batch_size, kv_len, self.n_heads, self.d_keys)
        values = values.contiguous().view(batch_size, kv_len, self.n_heads, self.d_values)

        # Rearrange for batch matrix multiplication
        queries = queries.permute(0, 2, 1, 3).contiguous().view(-1, q_len, self.d_queries)
        keys = keys.permute(0, 2, 1, 3).contiguous().view(-1, kv_len, self.d_keys)
        values = values.permute(0, 2, 1, 3).contiguous().view(-1, kv_len, self.d_values)

        # Attention scores
        attention_weights = torch.bmm(queries, keys.permute(0, 2, 1))
        attention_weights = (1.0 / math.sqrt(self.d_keys)) * attention_weights

        # Masking padding tokens
        not_pad_in_keys = (
            torch.LongTensor(range(kv_len))
            .unsqueeze(0)
            .unsqueeze(0)
            .expand_as(attention_weights)
            .to(DEVICE)
        )
        not_pad_in_keys = (
            not_pad_in_keys
            < key_value_sequence_lengths.repeat_interleave(self.n_heads)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand_as(attention_weights)
        )
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float("inf"))

        attention_weights = self.softmax(attention_weights)
        attention_weights = self.apply_dropout(attention_weights)

        # Apply attention to values
        sequences = torch.bmm(attention_weights, values)

        # Reshape back
        sequences = (
            sequences.contiguous()
            .view(batch_size, self.n_heads, q_len, self.d_values)
            .permute(0, 2, 1, 3)
        )
        sequences = sequences.contiguous().view(batch_size, q_len, -1)

        # Output projection
        sequences = self.cast_output(sequences)
        sequences = self.apply_dropout(sequences) + input_to_add

        return sequences

# inspired by: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation
class PositionWiseFCNetwork(nn.Module):
    
    def __init__(self, d_model, d_inner, dropout):
        super(PositionWiseFCNetwork, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_inner)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_inner, d_model)
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        input_to_add = sequences.clone()
        sequences = self.layer_norm(sequences)
        sequences = self.apply_dropout(self.relu(self.fc1(sequences)))
        sequences = self.fc2(sequences)
        sequences = self.apply_dropout(sequences) + input_to_add
        return sequences


class BoardEncoder(nn.Module):
    def __init__(self, vocab_sizes, d_model, n_heads, d_queries, d_values, 
                 d_inner, n_layers, dropout):
        super(BoardEncoder, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # Embeddings for board state components
        self.turn_embeddings = nn.Embedding(vocab_sizes["turn"], d_model)
        self.white_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["white_kingside_castling_rights"], d_model
        )
        self.white_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["white_queenside_castling_rights"], d_model
        )
        self.black_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["black_kingside_castling_rights"], d_model
        )
        self.black_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["black_queenside_castling_rights"], d_model
        )
        self.board_position_embeddings = nn.Embedding(
            vocab_sizes["board_position"], d_model
        )

        # Positional embeddings (5 castling/turn tokens + 64 board squares = 69)
        self.positional_embeddings = nn.Embedding(69, d_model)

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList(
            [self.make_encoder_layer() for i in range(n_layers)]
        )

        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def make_encoder_layer(self):
        encoder_layer = nn.ModuleList([
            MultiHeadAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_queries=self.d_queries,
                d_values=self.d_values,
                dropout=self.dropout,
                in_decoder=False,
            ),
            PositionWiseFCNetwork(
                d_model=self.d_model, 
                d_inner=self.d_inner, 
                dropout=self.dropout
            ),
        ])
        return encoder_layer

    def forward(self, turns, white_kingside_castling_rights, 
                white_queenside_castling_rights, black_kingside_castling_rights,
                black_queenside_castling_rights, board_positions):
        batch_size = turns.size(0)

        # Embed all board state components
        embeddings = torch.cat([
            self.turn_embeddings(turns),
            self.white_kingside_castling_rights_embeddings(white_kingside_castling_rights),
            self.white_queenside_castling_rights_embeddings(white_queenside_castling_rights),
            self.black_kingside_castling_rights_embeddings(black_kingside_castling_rights),
            self.black_queenside_castling_rights_embeddings(black_queenside_castling_rights),
            self.board_position_embeddings(board_positions),
        ], dim=1)

        # Add positional embeddings and scale
        boards = embeddings + self.positional_embeddings.weight.unsqueeze(0)
        boards = boards * math.sqrt(self.d_model)
        boards = self.apply_dropout(boards)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            boards = encoder_layer[0](
                query_sequences=boards,
                key_value_sequences=boards,
                key_value_sequence_lengths=torch.LongTensor([69] * batch_size).to(DEVICE),
            )
            boards = encoder_layer[1](sequences=boards)

        boards = self.layer_norm(boards)
        return boards


class ChessTransformer(nn.Module):
    def __init__(self, CONFIG):
        super(ChessTransformer, self).__init__()

        self.code = "EFT"

        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

        # Board encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Output heads for from/to square prediction
        self.from_squares = nn.Linear(self.d_model, 1)
        self.to_squares = nn.Linear(self.d_model, 1)

        self.init_weights()

    def init_weights(self):
        # Glorot uniform for most parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # Normal initialization for embeddings
        std = math.pow(self.d_model, -0.5)
        nn.init.normal_(self.board_encoder.board_position_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.board_encoder.turn_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.board_encoder.white_kingside_castling_rights_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.board_encoder.white_queenside_castling_rights_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.board_encoder.black_kingside_castling_rights_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.board_encoder.black_queenside_castling_rights_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.board_encoder.positional_embeddings.weight, mean=0.0, std=std)

    def forward(self, batch):
        
        # Encode board state
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
        )

        # Predict from/to squares (skip first 5 tokens which are castling/turn info)
        from_squares = self.from_squares(boards[:, 5:, :]).squeeze(2).unsqueeze(1)
        to_squares = self.to_squares(boards[:, 5:, :]).squeeze(2).unsqueeze(1)

        return from_squares, to_squares