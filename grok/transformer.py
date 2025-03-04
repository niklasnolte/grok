#!/usr/bin/env python
from typing import Tuple, List, Union

import torch
import torch.nn as nn
from numpy import cos, sin, sqrt
from torch import tensor, Tensor


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int):

        super().__init__()

        self.d_key = d_key

        # head projections
        self.Wq = nn.Linear(d_model, d_key, bias=False)
        self.Wk = nn.Linear(d_model, d_key, bias=False)
        self.Wv = nn.Linear(d_model, d_key, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Union[Tensor, None] = None,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:

        # project queries, keys, values
        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)

        # calculate compatibility function
        attn = torch.matmul(queries, torch.transpose(keys, -2, -1))
        attn = attn / sqrt(self.d_key)

        # Filter out attention to future positions
        if mask is not None:
            attn.masked_fill_(mask == 0, float("-inf"))

        # softmax
        attn = self.softmax(attn)

        # sum the weighted value vectors
        result: Tensor = torch.matmul(attn, values)  # shape = (max_context_len, d_key)

        return result


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        d_key = int(d_model / heads)

        attn_heads = [AttentionHead(d_model, d_key) for _ in range(heads)]
        self.attn_heads = nn.ModuleList(attn_heads)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor = None,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:

        head_outputs = [
            h(queries=queries, keys=keys, values=values, mask=mask,)
            for h in self.attn_heads
        ]

        multihead_result = torch.cat(head_outputs, dim=-1)
        multihead_result = self.Wo(multihead_result)
        return multihead_result


class DecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, heads: int, dropout: float, non_linearity: str = "relu",
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, heads)
        self.self_attn_norm = nn.LayerNorm(d_model)

        non_linearities = {"relu": nn.ReLU, "gelu": nn.GELU}
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            non_linearities[non_linearity](),
            nn.Linear(d_model * 4, d_model, bias=False),
        )
        if dropout == 0:
            self.ffn_drop = nn.Identity()
        else:
            self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, self_attn_mask: Tensor = None,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        a1 = self.self_attn(x, x, x, self_attn_mask)
        a1 = self.self_attn_norm(x + a1)

        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a1 = self.ffn_drop(a1)
        a2 = self.ffn_norm(a1 + a2)


        return a2


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        num_blocks: int,
        dropout: float,
        non_linearity: str = "relu",
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, heads, dropout, non_linearity)
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, x: Tensor, self_attn_mask: Tensor = None,
    ) -> Tuple[Tensor, List[List[Tensor]], List[List[Tensor]]]:
        for block in self.blocks:
            x = block(x, self_attn_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        non_linearity: str = "relu",
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.non_linearity = non_linearity

        self.vocab_len = vocab_len

        self.embedding = nn.Embedding(vocab_len, d_model)  # type: ignore
        self.register_buffer(
            "position_encoding", self._position_encoding(max_context_len, d_model)
        )
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len))

        self.decoder = Decoder(d_model, n_heads, n_layers, dropout, self.non_linearity,)

        self.linear = nn.Linear(d_model, vocab_len, bias=False)

    @staticmethod
    def make_mask(context_len: int) -> Tensor:
        return torch.ones([context_len, context_len]).tril()

    @classmethod
    def _position_encoding(cls, context_len: int, d_model: int) -> Tensor:
        rows = [
            tensor(
                [
                    sin(pos / (10000 ** (i / d_model)))
                    if i % 2 == 0
                    else cos(pos / (10000 ** ((i - 1) / d_model)))
                    for i in range(d_model)
                ]
            )
            for pos in range(context_len)
        ]
        stack = torch.stack(rows, dim=1)

        return stack.T  # type: ignore

    def embed(self, indices: Tensor) -> Tensor:
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len]  # type: ignore
        embedded = self.embedding(indices)
        return pe + embedded

    def forward(
        self, x: Tensor, pos: int = None,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """parameters:
        x:  (rank-1 tensor) vocab indices of decoder input token
                     sequence"""

        # Make sure sampling inputs are on the correct device
        x = x.to(self.embedding.weight.device)

        # make_attention mask
        this_max_context_len = x.shape[-1]
        self_attn_mask = self.self_attn_mask[  # type: ignore
            :this_max_context_len, :this_max_context_len
        ]

        # Decode
        x = self.embed(x)
        decoded = self.decoder(x, self_attn_mask)

        # Return predictions for specific token
        if pos is not None:
            decoded = decoded[:, pos, :]

        y_hat = self.linear(decoded)
        return y_hat
