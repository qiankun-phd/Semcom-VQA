"""Thin wrapper around the official SwinJSCC code (used as a black box, weights untouched).

Channel uses are counted as complex symbols: CBR = rate / (2*3*2^(2*downsample)) = rate/1536 for HR models,
so k = CBR * 3*H*W = rate*H*W/512 for a padded H x W input.
"""
import os, sys, types
import numpy as np
import torch
import torch.nn as nn

SJ = os.path.expanduser('~/phd_research/BUBBLES_hoverreport_20260930/SwinJSCC')
sys.path.insert(0, SJ)
from net.network import SwinJSCC  # noqa: E402

PAD = 128  # patch 2 x 3 merges -> /16, then window 8 -> input dims must be multiples of 128


def build(weight, model='SwinJSCC_w/_SAandRA', channel='awgn', C='32,64,96,128,192', snrs='1,4,7,10,13', size='base'):
    args = types.SimpleNamespace(model=model, channel_type=channel, C=C, multiple_snr=snrs, model_size=size,
                                 distortion_metric='MSE', trainset='DIV2K', testset='kodak', training=False)
    ch = None if model in ('SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA') else int(C)
    depths = {'small': [2, 2, 2, 2], 'base': [2, 2, 6, 2], 'large': [2, 2, 18, 2]}[size]
    common = dict(model=model, img_size=(256, 256), C=ch, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                  norm_layer=nn.LayerNorm, patch_norm=True)
    enc = dict(patch_size=2, in_chans=3, embed_dims=[128, 192, 256, 320], depths=depths, num_heads=[4, 6, 8, 10], **common)
    dec = dict(embed_dims=[320, 256, 192, 128], depths=depths[::-1], num_heads=[10, 8, 6, 4], **common)
    config = types.SimpleNamespace(encoder_kwargs=enc, decoder_kwargs=dec, logger=None, pass_channel=True, CUDA=True,
                                   device=torch.device('cuda:0'), norm=False, downsample=4)
    net = SwinJSCC(args, config)
    net.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)   # third-party file: tensors only
    return net.cuda().eval()


def _set_res(net, H, W):
    if (H, W) != (net.H, net.W):
        net.encoder.update_resolution(H, W)
        net.decoder.update_resolution(H // 16, W // 16)
        net.H, net.W = H, W


@torch.no_grad()
def transmit(net, x, snr_ch, snr_mod, rate, seed=None):
    """SA&RA model, one image per call (power normalisation is per call). snr_mod feeds the ModNets,
    snr_ch sets the AWGN noise; they differ only when we clamp the ModNet input outside the training range."""
    _, _, H, W = x.shape
    _set_res(net, H, W)
    if seed is not None:
        torch.manual_seed(seed)
    feat, mask = net.encoder(x, snr_mod, rate, net.model)
    avg_pwr = torch.sum(feat ** 2) / mask.sum()
    noisy = net.channel.forward(feat, snr_ch, avg_pwr) * mask
    rec = net.decoder(noisy, snr_mod, net.model)
    return rec.clamp(0., 1.), rate * H * W / 512


@torch.no_grad()
def transmit_fixed(net, x, snr, seed=None):
    """w/o SA&RA model at its own training point (sanity check only)."""
    _, _, H, W = x.shape
    if seed is not None:
        torch.manual_seed(seed)
    rec, cbr, _, _, _ = net(x, snr, int(net.channel_number[0]))
    return rec.clamp(0., 1.), cbr * 3 * H * W


def to_tensor(arr):
    """uint8 HxWx3 -> padded 1x3xH'xW' float cuda tensor (edge replicate) and the original (h, w)."""
    h, w = arr.shape[:2]
    H, W = -(-h // PAD) * PAD, -(-w // PAD) * PAD
    a = np.pad(arr, ((0, H - h), (0, W - w), (0, 0)), mode='edge')
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255.).cuda(), (h, w)


def to_uint8(rec, hw):
    h, w = hw
    return (rec[0, :, :h, :w].permute(1, 2, 0).cpu().numpy() * 255. + 0.5).astype(np.uint8)
