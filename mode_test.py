import torch
import torch.nn as nn


class PoseToTextTransformer(nn.Module):
    """
    Transformer encoder-decoder mapping per-frame pose features to text tokens.

    Inputs
    - feature: [B, T, D] float tensor (per-frame feature sequence)
    - feature_len: [B] long tensor (valid lengths per sample)
    - labels: [B, L] long tensor with BOS/EOS/PAD for teacher forcing

    Output
    - logits: [B, L-1, vocab_size] over next-token distribution
    """

    def __init__(
        self,
        feature_dim: int,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_src_len: int = 2048,
        max_tgt_len: int = 1024,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Frame embedding + positional encoding for encoder
        self.frame_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc_enc = nn.Parameter(torch.randn(1, max_src_len, d_model) * 0.01)

        # Token embedding + positional encoding for decoder
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc_dec = nn.Parameter(torch.randn(1, max_tgt_len, d_model) * 0.01)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def make_src_padding_mask(self, feature_len: torch.Tensor, T: int) -> torch.Tensor:
        """Return [B, T] bool mask where True marks padding positions."""
        # positions >= valid length are padding
        idxs = torch.arange(T, device=feature_len.device)[None, :]
        return idxs >= feature_len[:, None]

    def make_tgt_padding_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Return [B, L] bool mask for target padding tokens."""
        return labels.eq(self.pad_id)

    def make_causal_mask(self, size: int) -> torch.Tensor:
        """Return [size, size] bool upper-triangular mask (True blocks future positions)."""
        return torch.triu(torch.ones(size, size, device=self.pos_enc_enc.device), diagonal=1).bool()

    def forward(self, feature: torch.Tensor, feature_len: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, T, D = feature.shape
        B2, L = labels.shape
        assert B == B2, "Batch size mismatch between feature and labels"

        # Encoder inputs
        if T > self.max_src_len:
            raise ValueError(f"source length {T} exceeds max_src_len {self.max_src_len}")
        src = self.frame_proj(feature) + self.pos_enc_enc[:, :T, :]  # [B, T, d_model]
        src_key_padding = self.make_src_padding_mask(feature_len, T)  # [B, T]

        # Decoder inputs (teacher forcing)
        tgt_len = L - 1
        if tgt_len > self.max_tgt_len:
            raise ValueError(f"target length {tgt_len} exceeds max_tgt_len {self.max_tgt_len}")
        tgt_tokens = labels[:, :-1]  # [B, L-1]
        tgt = self.emb(tgt_tokens) + self.pos_enc_dec[:, :tgt_len, :]  # [B, L-1, d_model]
        tgt_key_padding = self.make_tgt_padding_mask(tgt_tokens)  # [B, L-1]

        # Causal mask prevents attending to future positions in decoder
        causal_mask = self.make_causal_mask(tgt_len)  # [L-1, L-1]

        out = self.transformer(
            src=src,
            tgt=tgt,
            src_key_padding_mask=src_key_padding,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=src_key_padding,
            tgt_mask=causal_mask,
        )  # [B, L-1, d_model]

        logits = self.out(out)  # [B, L-1, vocab_size]
        return logits


def build_model_from_dims(feature_dim: int, vocab_size: int, pad_id: int) -> PoseToTextTransformer:
    """Convenience builder with defaults aligned to the GRU baseline sizes."""
    return PoseToTextTransformer(
        feature_dim=feature_dim,
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    )
