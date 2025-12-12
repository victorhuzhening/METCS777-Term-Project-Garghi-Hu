import torch
import torch.nn as nn
from typing import Optional, Tuple



class PoseEncoder(nn.Module):
    """
    Encodes ASL feature vectors into latent states for decoder
    Layers in order:
      - Linear projection layer
      - 1D temporal conv
      - TransformerEncoder stack

    Input dimensions:
      x:      [B, T, D_in]
      length: [B]

    Output dimensions:
      enc_hidden: [B, T', d_model]
      pad_mask:   [B, T']
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 768,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_proj = nn.Linear(input_dim, d_model)

        # Temporal downsampling: stride=2 reduces sequence length
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:      [B, T, D_in]
        length: [B]
        returns:
          enc_hidden: [B, T', d_model]
          pad_mask:   [B, T'] (bool, True where PAD)
        """
        B, T, D = x.shape
        device = x.device

        x = self.in_proj(x)            # [B, T, d_model]

        # Build padding mask
        max_t = T
        pad_mask = (
            torch.arange(max_t, device=device)[None, :].expand(B, max_t)
            >= length[:, None]
        )                              # [B, T]

        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Convolute along time to learn temporal info
        x = x.transpose(1, 2)          # [B, d_model, T]
        x = self.conv(x)               # [B, d_model, T']
        x = x.transpose(1, 2)          # [B, T', d_model]
        T_prime = x.size(1)

        # Recompute lengths after conv: ceil(len/2)
        length_ds = (length + 1) // 2
        max_t_ds = T_prime
        pad_mask_ds = (
            torch.arange(max_t_ds, device=device)[None, :].expand(B, max_t_ds)
            >= length_ds[:, None]
        )                              # [B, T']

        enc_hidden = self.encoder(
            x,
            src_key_padding_mask=pad_mask_ds,
        )                              # [B, T', d_model]

        return enc_hidden, pad_mask_ds



class EncoderDecoderModel(nn.Module):
    """
    Encoder-Decoder model for pose-2-text generative modelling.

    Blocks in order:
    Encoder: PoseEncoder over pose feature sequences
    Decoder: TransformerDecoder over target tokens
    Linear Layer: projects decoder states to vocab logits for translation

    At inference time we:
      - Encode pose once
      - Autoregressively call greedy_decode() to generate tokens.
    """
    def __init__(
        self,
        feature_dim: int,
        vocab_size: int,
        d_model: int = 768,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_tgt_len: int = 128,
        pad_id: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model
        self.max_tgt_len = max_tgt_len

        # Encoder over pose features vectors
        self.pose_encoder = PoseEncoder(
            input_dim=feature_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Token + position embeddings for decoder text
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_tgt_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def _build_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        Returns [L, L] bool mask where True indicates a masked position.
        Output used as target sequence for decoder
        """
        mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def encode_pose(
        self,
        pose_feats: torch.Tensor,
        pose_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one pose encoder block.
        """
        enc_hidden, enc_pad_mask = self.pose_encoder(pose_feats, pose_len)
        return enc_hidden, enc_pad_mask

    def forward(
        self,
        pose_feats: torch.Tensor,       # [B, T, D_in]
        pose_len: torch.Tensor,         # [B]
        decoder_input_ids: torch.Tensor # [B, L] (shifted-right tokens)
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing (gt labels are forced into model as input for next step)

        Params:
            pose_feats: [B, T, D_in]
            pose_len: [B]
            decoder_input_ids: [B, L]

        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = decoder_input_ids.shape
        device = decoder_input_ids.device

        # 1. Encoder
        enc_hidden, enc_pad_mask = self.encode_pose(pose_feats, pose_len)

        # 2. Decoder embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        tgt_emb = self.tok_embed(decoder_input_ids) + self.pos_embed(positions)     # [B, L, H]

        # 3. Causal and target padding masks
        causal_mask = self._build_causal_mask(L, device)                            # [L, L]
        tgt_pad_mask = (decoder_input_ids == self.pad_id)                           # [B, L]

        # 4. Decoder
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=enc_hidden,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=enc_pad_mask,
        )                                                                           # [B, L, H]

        # 5) Project output to vocab logits
        logits = self.lm_head(dec_out)                                           # [B, L, vocab_size]
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        pose_feats: torch.Tensor,
        pose_len: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode a simgle sample greedily
        Assumes bos_id and eos_id correspond to 1 and 2 in vocab.
        """
        self.eval()
        device = pose_feats.device
        max_len = max_len or self.max_tgt_len

        if self.bos_id is None or self.eos_id is None:
            raise ValueError(
                "greedy_decode requires bos_id and eos_id to be set on the model."
            )

        # Encode pose once
        enc_hidden, enc_pad_mask = self.encode_pose(pose_feats, pose_len)

        generated = [self.bos_id]

        for step in range(max_len):
            tgt_ids = torch.tensor(
                [generated],
                dtype=torch.long,
                device=device,
            )
            B, L = tgt_ids.shape

            positions = torch.arange(L, device=device).unsqueeze(0)
            tgt_emb = self.tok_embed(tgt_ids) + self.pos_embed(positions)

            causal_mask = self._build_causal_mask(L, device)
            tgt_pad_mask = (tgt_ids == self.pad_id)

            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=enc_hidden,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=enc_pad_mask,
            )

            logits = self.lm_head(dec_out[:, -1, :])
            next_id = int(logits.argmax(dim=-1).item())    # choose using greedy
            generated.append(next_id)

            if next_id == self.eos_id:
                break

        return torch.tensor(generated, dtype=torch.long, device=device)
