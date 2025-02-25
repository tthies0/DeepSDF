#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        class_embedding=False,
        use_transformers=False,
        transformer_hidden_size=1024,
        num_heads=16,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        self.input_coord_length = 3
        if class_embedding:
            self.input_coord_length = 12
        
        dims = [latent_size + self.input_coord_length] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            setattr(
                self,
                "multi_head_attn" + str(layer),
                nn.MultiheadAttention()
            )

            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.input_coord_length

            if use_transformers and (layer+1)%2 == 0:
                setattr(self, "transformer" + str(layer), TransformerLayer(dims[layer], out_dim, transformer_hidden_size, num_heads, dropout_prob))

            if use_transformers and (layer+1)%2 == 0:
                setattr(self, "transformer" + str(layer), TransformerLayer(dims[layer], out_dim, transformer_hidden_size, num_heads, dropout_prob))

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+4)
    def forward(self, input):
        xyz = input[:, -self.input_coord_length:]

        if input.shape[1] > 4 and self.latent_dropout:
            latent_vecs = input[:, :-self.input_coord_length]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer), None)
            transformer = getattr(self, "transformer" + str(layer), None)
            assert lin is not None or transformer is not None
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            if lin is not None:
                x = lin(x)
            else:
                x = transformer(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_layers,
            num_heads,
            dropout_prob=0.0,
    ):
        super(TransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout_prob)
        self.norm = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_layers),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_layers, output_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, input):
        x = input.unsqueeze(0) #Add fake sequence dim
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(input+attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x+ff_output)

        return x.squeeze(0)