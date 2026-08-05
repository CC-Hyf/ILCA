import gc
import os
import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()
        half_dim = time_emb_dims // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        ts = torch.arange(total_time_steps, dtype=torch.float32)
        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)

class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)
        h, _ = self.mhsa(h, h, h)
        h = h.swapaxes(2, 1).view(B, self.channels, H, W)
        return x + h


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_fn = nn.SiLU()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):

        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)
        h += self.dense_1(self.act_fn(t))[:, :, None, None]
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)
        h = h + self.match_input(x)
        h = self.attention(h)
        return h

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, h1, h2):

        B, C, H, W = h1.shape
        h1 = h1.view(B, C, H * W).permute(0, 2, 1)
        h2 = h2.view(B, C, H * W).permute(0, 2, 1)
        attn_output, _ = self.multihead_attn(h1, h2, h2)
        h2_out = self.norm(h1 + self.dropout(attn_output))
        h2_out = h2_out.permute(0, 2, 1).view(B, C, H, W)
        return h2_out


class UNet(nn.Module):
    def __init__(
            self,
            input_channels = 2,
            output_channels = 2,
            num_res_blocks = 2,
            base_channels = 8,
            base_channels_multiples = (1, 2, 4, 8),
            apply_attention = (False, False, True, False),
            dropout_rate = 0.1,
            time_multiple = 4,
    ):
        super().__init__()

        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)
        self.first_h1 = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")
        self.first_h2 = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")
        num_resolutions = len(base_channels_multiples)
        self.encoder_blocks_h1 = nn.ModuleList()
        self.encoder_blocks_h2 = nn.ModuleList()
        self.cross_attention_layers_enc = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(num_res_blocks):
                block_h1 = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                block_h2 = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                self.encoder_blocks_h1.append(block_h1)
                self.encoder_blocks_h2.append(block_h2)


                self.cross_attention_layers_enc.append(CrossAttention(embed_dim=out_channels))

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                self.encoder_blocks_h1.append(DownSample(channels=in_channels))
                self.encoder_blocks_h2.append(DownSample(channels=in_channels))

                curr_channels.append(in_channels)


        self.bottleneck_blocks_h1 = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                ),
            )
        )

        self.bottleneck_blocks_h2 = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                ),
            )
        )
        self.cross_attention_bottleneck = CrossAttention(embed_dim=in_channels)


        self.decoder_blocks_h1 = nn.ModuleList()
        self.decoder_blocks_h2 = nn.ModuleList()
        self.cross_attention_layers_dec = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]


            for _ in range(num_res_blocks+1):
                encoder_in_channels = curr_channels.pop()
                block_h1 = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                block_h2 = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                in_channels = out_channels
                self.decoder_blocks_h1.append(block_h1)
                self.decoder_blocks_h2.append(block_h2)


                self.cross_attention_layers_dec.append(CrossAttention(embed_dim=out_channels))


            if level != 0:
                self.decoder_blocks_h1.append(UpSample(in_channels))
                self.decoder_blocks_h2.append(UpSample(in_channels))
                self.final = nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, h1, h2, t):
        time_emb = self.time_embeddings(t)
        h1 = self.first_h1(h1)
        h2 = self.first_h2(h2)

        outs_h1 = [h1]
        outs_h2 = [h2]
        cross_attention_index = 0
        for i, (block_h1, block_h2) in enumerate(zip(self.encoder_blocks_h1, self.encoder_blocks_h2)):

            if isinstance(block_h1, ResnetBlock) and isinstance(block_h2, ResnetBlock):
                h1 = block_h1(h1, time_emb)
                h2 = block_h2(h2, time_emb)
                h2 = self.cross_attention_layers_enc[cross_attention_index](h1, h2)
                cross_attention_index += 1
            else:

                h1 = block_h1(h1)
                h2 = block_h2(h2)
            outs_h1.append(h1)
            outs_h2.append(h2)

        for layer1, layer2 in zip(self.bottleneck_blocks_h1, self.bottleneck_blocks_h2):
            h1 = layer1(h1, time_emb)
            h2 = layer2(h2, time_emb)
            h2 = self.cross_attention_bottleneck(h1, h2)
            cross_attention_index = 0

        for i, (block_h1, block_h2) in enumerate(zip(self.decoder_blocks_h1, self.decoder_blocks_h2)):
            if isinstance(block_h1, ResnetBlock):
                out_h1 = outs_h1.pop()
                h1 = torch.cat([h1, out_h1], dim=1)

            if isinstance(block_h2, ResnetBlock):
                out_h2 = outs_h2.pop()
                h2 = torch.cat([h2, out_h2], dim=1)


            h1 = block_h1(h1, time_emb)
            h2 = block_h2(h2, time_emb)


            if isinstance(block_h1, ResnetBlock) and isinstance(block_h2, ResnetBlock):
                h2 = self.cross_attention_layers_dec[cross_attention_index](h1, h2)
                cross_attention_index += 1

        h2 = self.final(h2)
        return h2

