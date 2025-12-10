import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
#  ECA – Enhanced Channel Attention for 1D
# -------------------------
class ECA1D(nn.Module):
    """
    Enhanced Channel Attention for 1D feature maps.
    Expects input: [B, C, T]
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        # x: [B, C, T]
        # Global average pooling over time: [B, C, 1]
        y = x.mean(dim=-1, keepdim=True)       # [B, C, 1]

        # Treat channels as "time" for conv: [B, 1, C] -> conv -> [B, 1, C]
        y = y.transpose(1, 2)                  # [B, 1, C]
        y = self.conv(y)                       # [B, 1, C]
        y = y.transpose(1, 2)                  # [B, C, 1]

        y = torch.sigmoid(y)                   # [B, C, 1]
        return x * y                           # broadcast over T


# -------------------------
#  Depthwise Conv1d
# -------------------------
class DepthwiseConv1d(nn.Module):
    """
    Depthwise 1D convolution (groups = in_channels).
    Expects input: [B, C, T]
    """
    def __init__(self, in_channels, kernel_size, stride=1, dilation=1, bias=False):
        super().__init__()
        # "same" padding approximation
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


# -------------------------
#  Conv1DBlock (TF Conv1DBlock analog)
# -------------------------
class Conv1DBlock(nn.Module):
    """
    Efficient Conv1D block with:
      1x1 expansion -> depthwise conv -> BN -> ECA -> 1x1 projection -> Dropout -> Residual

    Expects input: [B, C, T] with C == channel_size.
    """
    def __init__(
        self,
        channel_size,        # C_out = C_in
        kernel_size,
        dilation_rate=1,
        drop_rate=0.0,
        expand_ratio=2,
        activation="tanh",
    ):
        super().__init__()
        self.channel_size = channel_size
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            # fall back to tanh
            self.act = torch.tanh

        in_channels = channel_size
        hidden_channels = in_channels * expand_ratio

        # 1x1 "Dense" expansion: C_in -> hidden_channels
        self.expand = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=True)

        # Depthwise temporal conv
        self.dwconv = DepthwiseConv1d(
            in_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(hidden_channels, momentum=0.95)
        self.eca = ECA1D(kernel_size=5)

        # Projection back to channel_size
        self.project = nn.Conv1d(hidden_channels, channel_size, kernel_size=1, bias=True)

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, C, T]
        skip = x

        x = self.expand(x)          # [B, hidden, T]
        x = self.act(x)
        x = self.dwconv(x)          # [B, hidden, T]
        x = self.bn(x)
        x = self.eca(x)             # [B, hidden, T]
        x = self.project(x)         # [B, C, T]
        x = self.dropout(x)

        if x.shape == skip.shape:
            x = x + skip            # residual

        return x


# -------------------------
#  Positional Encoding (sin/cos)
# -------------------------
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds [1, max_len, dim] to input [B, T, dim].
    """
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, dim]
        T = x.size(1)
        return x + self.pe[:, :T]


# -------------------------
#  TransformerBlock (encoder style)
# -------------------------
class TransformerBlock(nn.Module):
    """
    Encoder-style transformer block with:
      LN -> MHA -> Dropout -> Residual
      LN -> FFN -> Dropout -> Residual

    Expects input: [B, T, dim]
    key_padding_mask: [B, T] with True for padded positions.
    """
    def __init__(
        self,
        dim=256,
        num_heads=4,
        expand=4,
        attn_dropout=0.2,
        drop_rate=0.2,
        activation="gelu",
    ):
        super().__init__()
        self.dim = dim
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # inputs: [B, T, dim]
        )
        self.dropout1 = nn.Dropout(drop_rate)
        self.norm1 = nn.LayerNorm(dim)

        if activation == "gelu":
            act_layer = nn.GELU
        elif activation == "relu":
            act_layer = nn.ReLU
        else:
            act_layer = nn.GELU

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand, bias=False),
            act_layer(),
            nn.Linear(dim * expand, dim, bias=False),
        )
        self.dropout2 = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, dim]
        # key_padding_mask: [B, T] with True for PAD positions
        attn_out, _ = self.mha(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        return x


# -------------------------
#  KeypointCTCModel – drop-in for your ASLData features
# -------------------------
class KeypointCTCModel(nn.Module):
    """
    Drop-in model for your ASLData pipeline.

    Expects:
      features:    [B, T, D]  (from asl_collate_func)
      feature_len: [B]        (valid lengths per sample)

    Returns:
      logits_ctc:   [T, B, vocab_size] – ready for nn.CTCLoss
      input_lengths: [B]      – same as feature_len
    """
    def __init__(
        self,
        input_dim,           # D from your feature vector
        vocab_size,
        dim=256,             # internal model dimension / conv channels
        num_conv_blocks=3,
        num_transformer_blocks=3,
        max_len=5000,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Project raw features to model dim
        self.input_proj = nn.Linear(input_dim, dim, bias=False)

        # Positional encoding
        self.pos_enc = PositionalEncoding(dim=dim, max_len=max_len)

        # Conv/ECA frontend (in [B, C, T])
        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(
                channel_size=dim,
                kernel_size=11,
                dilation_rate=1,
                drop_rate=0.1,
                expand_ratio=2,
                activation="tanh",
            )
            for _ in range(num_conv_blocks)
        ])

        # Transformer encoder stack
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=4,
                expand=4,
                attn_dropout=0.2,
                drop_rate=0.2,
                activation="gelu",
            )
            for _ in range(num_transformer_blocks)
        ])

        # Final classification head (per timestep)
        self.classifier = nn.Linear(dim, vocab_size, bias=True)

    def forward(self, features, feature_len):
        """
        features:    [B, T, D]
        feature_len: [B]  (valid time steps per sequence)

        Returns:
          logits_ctc:    [T, B, vocab_size]
          input_lengths: [B]
        """
        # Project to model dim
        x = self.input_proj(features)   # [B, T, dim]

        # Positional encoding
        x = self.pos_enc(x)            # [B, T, dim]

        # Conv frontend in [B, C, T]
        x = x.transpose(1, 2)          # [B, dim, T]
        for block in self.conv_blocks:
            x = block(x)               # [B, dim, T]
        x = x.transpose(1, 2)          # [B, T, dim]

        # Build key_padding_mask for transformer (True where PAD)
        B, T, _ = x.shape
        device = x.device
        seq_range = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
        key_padding_mask = seq_range >= feature_len.unsqueeze(1)              # [B, T]

        # Transformer encoder stack
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)  # [B, T, dim]

        # Time-distributed classification
        logits = self.classifier(x)    # [B, T, vocab_size]

        # For CTCLoss, need [T, B, C]
        logits_ctc = logits.transpose(0, 1)  # [T, B, vocab_size]
        input_lengths = feature_len          # [B]

        return logits_ctc, input_lengths
