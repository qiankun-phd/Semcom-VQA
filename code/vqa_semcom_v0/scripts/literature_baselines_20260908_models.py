"""Auditable VisDrone adaptations; not claimed as original-paper reproductions."""
from __future__ import annotations
from functools import partial
from pathlib import Path
import sys
import torch
from torch import nn
from transformers import BertModel


class TDeepSCVisDrone(nn.Module):
    """Preserve upstream factory's core encoders/decoder; relocate question locally."""
    def __init__(self, root: Path, answers: int, complex_symbols: int = 24):
        super().__init__()
        sys.path.insert(0, str(root / 'upstream/t-udeepsc/TDeepSC'))
        from model_util import ViTEncoder_vqa
        from trans_decoder import Decoder
        assert complex_symbols in (24, 48, 96)
        self.complex_symbols = complex_symbols
        dimensions = 2 * complex_symbols // 3
        self.img_encoder = ViTEncoder_vqa(embed_dim=384, depth=4, num_heads=6, mlp_ratio=4,
                                         qkv_bias=True, init_values=0., norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.text_encoder = BertModel.from_pretrained(root / 'weights_bert/bert-small', local_files_only=True)
        assert self.text_encoder.config.hidden_size == 512
        self.img_encoder_to_channel = nn.Linear(384, dimensions)
        self.text_encoder_to_channel = nn.Linear(512, 6)
        self.img_channel_to_decoder = nn.Linear(dimensions, 128)
        self.text_channel_to_decoder = nn.Linear(6, 128)
        self.decoder = Decoder(depth=2, embed_dim=128, num_heads=4, dff=512, drop_rate=0.)
        self.query_embedd = nn.Embedding(25, 128)
        self.head = nn.Linear(128, answers)

    @staticmethod
    def rician(x: torch.Tensor, snr: torch.Tensor, noiseless: bool = False, channel_seeds: list[int] | None = None) -> torch.Tensor:
        shape = x.shape
        z = torch.view_as_complex(x.float().reshape(shape[0], -1, 2).contiguous())
        z = z / z.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        if not noiseless:
            k = 10 ** .6
            if channel_seeds is None:
                fades = torch.randn(shape[0], 2, device=x.device)
                noise_parts = torch.randn(shape[0], z.shape[1], 2, device=x.device)
            else:
                generators = [torch.Generator(device=x.device).manual_seed(seed) for seed in channel_seeds]
                fades = torch.stack([torch.randn(2, device=x.device, generator=g) for g in generators])
                noise_parts = torch.stack([torch.randn(z.shape[1], 2, device=x.device, generator=g) for g in generators])
            h = (k / (k + 1)) ** .5 + torch.view_as_complex(fades.contiguous())[:, None] / (2 * (k + 1)) ** .5
            noise = torch.view_as_complex(noise_parts.contiguous()) * (10 ** (-snr[:, None] / 20)) / 2 ** .5
            z = (h * z + noise) / h
        return torch.view_as_real(z).reshape(shape).to(x.dtype)

    def forward(self, image: torch.Tensor, tokens: dict, snr: torch.Tensor, noiseless: bool = False, channel_seeds: list[int] | None = None) -> torch.Tensor:
        x = self.img_encoder(image, 'vqa')[:, :3]
        x = self.rician(self.img_encoder_to_channel(x), snr, noiseless, channel_seeds)
        x = self.img_channel_to_decoder(x)
        question = self.text_encoder(**tokens).last_hidden_state[:, :2]
        question = self.text_channel_to_decoder(self.text_encoder_to_channel(question))
        memory = torch.cat([x, question], 1)
        query = self.query_embedd.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        decoded = self.decoder(query, memory, None, None, None)
        return self.head(decoded.mean(1))


class RSVQAHead(nn.Module):
    """Original512px visual projection,1200-D product fusion,256-D answer head."""
    def __init__(self, answers: int):
        super().__init__()
        self.visual_compress = nn.Conv2d(2048, 8, 1)
        self.linear_v = nn.Linear(2048, 1200)
        self.linear_q = nn.Linear(2400, 1200)
        self.linear_classif1 = nn.Linear(1200, 256)
        self.linear_classif2 = nn.Linear(256, answers)
        self.dropout = nn.Dropout(.5)

    def forward(self, image: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        visual = self.visual_compress(image).flatten(1)
        visual = torch.tanh(self.linear_v(self.dropout(visual)))
        question = torch.tanh(self.linear_q(self.dropout(question)))
        fused = self.dropout(torch.tanh(visual * question))
        return self.linear_classif2(self.dropout(torch.tanh(self.linear_classif1(fused))))
