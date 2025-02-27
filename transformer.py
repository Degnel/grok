from attention import MultiHeadAttention
import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Implements a Transformer model with configurable depth and optional quantization
    for various components.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        n_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the feedforward network.
        depth (int): The number of encoder layers in the Transformer.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        vocab_size (int, optional): The size of the input vocabulary. If None, no embedding layer is added. Defaults to None.
        max_context_size (int, optional): The maximum length of the input sequences. Defaults to 512.
        mask (bool, optional): If True, adds a mask to the attention scores. Defaults to False.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        depth: int,
        dropout: float = 0.1,
        vocab_size: int | None = None,
        max_context_size: int = 512,
        mask: bool = False,
        lora_ratio: float = 4
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.mask = mask

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    lora_ratio,
                )
                for _ in range(depth)
            ]
        )

        if vocab_size:
            self.embedding = nn.Embedding(
                vocab_size + 1, d_model
            )
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
            self.output_projection.weight = self.embedding.weight
            self.position_embedding = nn.Embedding(max_context_size, d_model)
        else:
            self.embedding = None
            self.output_projection = None
            self.position_embedding = None

    def forward(self, x: torch.Tensor):
        """
        x : Tensor de taille (batch_size, seq_len) ou (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        if self.mask:
            mask = torch.triu(torch.ones(seq_len, seq_len)).bool().to(x.device)
        else:
            mask = None

        if self.embedding is not None:
            x = x.to(torch.int32)
            x = (
                self.embedding(x) + self.position_embedding.weight
            )  # [batch_size, seq_len, d_model]

        for layer in self.encoder_layers:
            x = layer(x, mask)

        if self.output_projection is not None:
            x = x - self.position_embedding.weight
            x = self.output_projection(x)

        return x.transpose(1, 2)

class TransformerEncoderLayer(nn.Module):
    """
    Args:
        d_model (int): The dimensionality of the input embeddings.
        n_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the feedforward network.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        lora_ratio: float = 4,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model, n_heads, lora_ratio
        )

        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)

        self.activation = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.previous_weights = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        mask : Tensor de taille (seq_len, seq_len)
        """
        # Attention multi-têtes
        attn_output = self.self_attention(x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # Réseau feed-forward
        ff_output = self.fc_2(self.activation(self.fc_1(x)))
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x