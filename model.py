import torch
import torch.nn as nn


class PoseToTextModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,          # D: feature dimension of feature per frame
        enc_hidden: int,        # encoder GRU hidden size per direction
        vocab_size: int,        # |V|
        emb_dim: int,           # token embedding dim
        pad_id: int,
        num_enc_layers: int = 1,
        num_dec_layers: int = 1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size

        # Encoder: Bi-GRU over feature sequence
        self.encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=enc_hidden,
            num_layers=num_enc_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Decoder embedding
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_id,
        )

        # Decoder GRU: hidden size = 2 * enc_hidden (concat directions)
        self.decoder = nn.GRU(
            input_size=emb_dim,
            hidden_size=2 * enc_hidden,
            num_layers=num_dec_layers,
            batch_first=True,
        )

        # Final projection to vocab
        self.out = nn.Linear(2 * enc_hidden, vocab_size)

    def encode(self, feature, feature_len):
        """
        feature: [B, T, D]
        feature_len: [B]
        Returns: encoder final hidden state [num_layers*2, B, H]
        """
        # Pack for efficient RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            feature,
            lengths=feature_len.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        enc_out, h_n = self.encoder(packed)
        # h_n: [num_layers*2, B, enc_hidden]
        return h_n

    def forward(self, feature, feature_len, labels):
        """
        feature:   [B, T, D]
        feature_len: [B]
        labels: [B, L]  (with <bos> ... <eos> and <pad>)

        We use teacher forcing:
          decoder inputs: labels[:, :-1]
          targets:        labels[:, 1:]
        Returns:
          logits: [B, L-1, vocab_size]
        """
        B, T, D = feature.shape
        B2, L = labels.shape
        assert B == B2

        # ---- Encode ----
        h_n = self.encode(feature, feature_len)  # [num_layers*2, B, enc_hidden]

        # Merge directions for final layer into a single initial hidden state
        # For simplicity, we only use last layer’s forward/backward
        # h_n_last: [2, B, enc_hidden] -> concat -> [1, B, 2*enc_hidden]
        num_layers_times_dir, B_enc, H = h_n.shape
        assert B_enc == B
        h_n_last = h_n[-2:]                 # [2, B, H] (last layer forward/backward)
        h0_dec = torch.cat(
            [h_n_last[0], h_n_last[1]], dim=-1
        ).unsqueeze(0)                      # [1, B, 2H]

        # ---- Decode with teacher forcing ----
        # decoder input is labels shifted right (all but last token)
        dec_inp = labels[:, :-1]            # [B, L-1]
        emb = self.emb(dec_inp)             # [B, L-1, emb_dim]

        dec_out, _ = self.decoder(emb, h0_dec)  # [B, L-1, 2H]
        logits = self.out(dec_out)              # [B, L-1, vocab_size]

        return logits
