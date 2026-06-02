"""RGB-guided depth upsampling network.

Architecture:
  - RGB encoder: pretrained MobileNetV2 with feature taps at strides 2/4/8/16
  - Depth encoder: lightweight CNN over the bicubic-upsampled lowres depth
    (+ confidence channel) at strides 1/2/4/8/16
  - Decoder: progressive bilinear upsampling that fuses RGB and depth features
    at every scale (concat + 3x3 conv blocks)
  - Output: scalar residual added to the bicubic baseline (in normalized space)

The choice of MobileNetV2 (over ResNet) follows the spec — it converts cleanly
to CoreML for iPhone deployment while still providing strong edge guidance.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class RGBEncoder(nn.Module):
    """MobileNetV2 wrapper that exposes 4 multi-scale feature maps."""

    # taps after these MobileNetV2 stages (output strides):
    #   features[1]  ->  16 ch, stride 2
    #   features[3]  ->  24 ch, stride 4
    #   features[6]  ->  32 ch, stride 8
    #   features[13] ->  96 ch, stride 16
    TAP_INDICES = (1, 3, 6, 13)
    TAP_CHANNELS = (16, 24, 32, 96)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mnv2 = mobilenet_v2(weights=weights)
        self.stem = mnv2.features[0]
        self.stage1 = mnv2.features[1]                                     # stride 2, 16 ch  (tap 0)
        self.stage2 = nn.Sequential(mnv2.features[2], mnv2.features[3])    # stride 4, 24 ch  (tap 1)
        self.stage3 = nn.Sequential(*[mnv2.features[i] for i in range(4, 7)])    # stride 8, 32 ch (tap 2)
        self.stage4 = nn.Sequential(*[mnv2.features[i] for i in range(7, 14)])   # stride 16, 96 ch (tap 3)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)   # stride 2,  16
        f2 = self.stage2(f1)  # stride 4,  24
        f3 = self.stage3(f2)  # stride 8,  32
        f4 = self.stage4(f3)  # stride 16, 96
        return [f1, f2, f3, f4]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()


class DepthEncoder(nn.Module):
    """Lightweight CNN over depth + confidence channels at the target res.

    channels[i] = output width of the i-th downsampling stage (0-indexed).
    Stem outputs channels[0]. down_layers[0] keeps channels[0] width (stride 2);
    down_layers[i] for i>=1 goes channels[i-1]->channels[i].
    forward() returns levels+1 feature maps at strides 1, 2, 4, ..., 2^levels.
    """

    def __init__(
        self,
        in_ch: int = 2,
        levels: int = 4,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        filter_size: int = 3,
    ):
        super().__init__()
        assert len(channels) == levels, f"channels must have {levels} entries, got {len(channels)}"
        self.levels = levels
        self.channels = tuple(channels)
        k = filter_size

        self.stem = _conv_bn_relu(in_ch, channels[0], k=k)
        self.down_layers = nn.ModuleList()
        # down[0]: channels[0] -> channels[0], stride 2
        self.down_layers.append(
            nn.Sequential(_conv_bn_relu(channels[0], channels[0], k=k), nn.MaxPool2d(2))
        )
        # down[i] for i=1..levels-1: channels[i-1] -> channels[i], stride 2
        for i in range(1, levels):
            self.down_layers.append(
                nn.Sequential(_conv_bn_relu(channels[i - 1], channels[i], k=k), nn.MaxPool2d(2))
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.stem(x)]
        for dl in self.down_layers:
            feats.append(dl(feats[-1]))
        return feats  # length = levels + 1, strides 1, 2, 4, ..., 2^levels

    def feat_ch(self, idx: int) -> int:
        """Output channels at depth_feats[idx]: idx=0 -> channels[0], idx>=1 -> channels[idx-1]."""
        return self.channels[0] if idx == 0 else self.channels[idx - 1]


class FuseUpBlock(nn.Module):
    """Bilinear-upsample + concat (RGB tap, depth tap, prev decoder feat) + 2x conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            _conv_bn_relu(in_ch, out_ch, k=3),
            _conv_bn_relu(out_ch, out_ch, k=3),
        )

    def forward(self, x: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
        target_size = skips[0].shape[-2:]
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = torch.cat([x] + list(skips), dim=1)
        return self.fuse(x)


_HEAD_CH = 16  # fixed width of the final pre-head feature map


class RGBGuidedDepthUpsampler(nn.Module):
    """End-to-end model: RGB+depth -> residual -> normalized depth output."""

    def __init__(
        self,
        pretrained_rgb: bool = True,
        residual_scale: float = 0.2,
        depth_encoder_levels: int = 4,
        depth_filter_size: int = 3,
        depth_channels: Tuple[int, ...] = (32, 64, 128, 256),
        fusion_decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()
        assert len(depth_channels) == depth_encoder_levels
        self.residual_scale = residual_scale
        self._levels = depth_encoder_levels

        self.rgb_enc = RGBEncoder(pretrained=pretrained_rgb)
        self.depth_enc = DepthEncoder(
            in_ch=2,
            levels=depth_encoder_levels,
            channels=depth_channels,
            filter_size=depth_filter_size,
        )

        levels = depth_encoder_levels
        rgb_chs = RGBEncoder.TAP_CHANNELS  # (16, 24, 32, 96) at strides 2,4,8,16
        d_ch = self.depth_enc.feat_ch       # callable: depth_feats[idx] -> channel count

        # use first `levels` entries of fusion_decoder_channels
        fdc = list(fusion_decoder_channels)[:levels]

        # bottom block at deepest stride: concat(d[levels], rgb[levels-1]) -> fdc[0]
        self.bot = _conv_bn_relu(d_ch(levels) + rgb_chs[levels - 1], fdc[0], k=3)

        # up_blocks[i] for i=0..levels-2: upsample and fuse with next RGB tap + depth skip
        # block i: prev=fdc[i], rgb=rgb[levels-2-i], depth=d[levels-1-i] -> fdc[i+1]
        self.up_blocks = nn.ModuleList()
        for i in range(levels - 1):
            rgb_idx = levels - 2 - i
            d_idx = levels - 1 - i
            in_ch = fdc[i] + rgb_chs[rgb_idx] + d_ch(d_idx)
            self.up_blocks.append(FuseUpBlock(in_ch, fdc[i + 1]))

        # final up to full resolution: fuse with depth stem only
        self.up0 = FuseUpBlock(fdc[levels - 1] + d_ch(0), _HEAD_CH)

        self.head = nn.Sequential(
            _conv_bn_relu(_HEAD_CH, _HEAD_CH, k=3),
            nn.Conv2d(_HEAD_CH, 1, 3, padding=1),
        )
        # init head weights to zero: model starts as identity (outputs bicubic baseline)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        rgb: torch.Tensor,
        bicubic_norm: torch.Tensor,
        conf_hi: torch.Tensor,
    ) -> torch.Tensor:
        """Returns predicted depth in *normalized* units (depth / DEPTH_MAX)."""
        depth_in = torch.cat([bicubic_norm, conf_hi], dim=1)
        rgb_feats = self.rgb_enc(rgb)           # [f1..f4] at strides 2,4,8,16
        depth_feats = self.depth_enc(depth_in)  # [d0..d_levels] at strides 1,2,...,2^levels

        levels = self._levels
        b = self.bot(torch.cat([depth_feats[levels], rgb_feats[levels - 1]], dim=1))

        x = b
        for i, blk in enumerate(self.up_blocks):
            rgb_idx = levels - 2 - i
            d_idx = levels - 1 - i
            x = blk(x, rgb_feats[rgb_idx], depth_feats[d_idx])

        u0 = self.up0(x, depth_feats[0])
        residual = self.head(u0)
        return bicubic_norm + self.residual_scale * residual

    def freeze_rgb(self):
        self.rgb_enc.freeze()

    def unfreeze_rgb(self):
        self.rgb_enc.unfreeze()


def count_parameters(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable
