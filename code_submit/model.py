from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

#conv2d, batchnorm, ReLu. padding applied automatically
def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


#feature extractor for highres RGB image (mobilenetv2 at 4 spatial scales)
class RGBEncoder(nn.Module):
    # strides 2/4/8/16 → 16/24/32/96 channels
    TAP_INDICES = (1, 3, 6, 13)
    TAP_CHANNELS = (16, 24, 32, 96)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mnv2 = mobilenet_v2(weights=weights)
        self.stem = mnv2.features[0]
        self.stage1 = mnv2.features[1] # stride 2, 16 ch (tap 0)
        self.stage2 = nn.Sequential(mnv2.features[2], mnv2.features[3]) # stride 4, 24 ch (tap 1)
        self.stage3 = nn.Sequential(*[mnv2.features[i] for i in range(4, 7)]) # stride 8, 32 ch (tap 2)
        self.stage4 = nn.Sequential(*[mnv2.features[i] for i in range(7, 14)]) # stride 16, 96 ch (tap 3)

    #pass rgb image thru network, return feature maps at 4 scales
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

    #freeze mobilenetv2
    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
    
    #unfreeze mobilenetv2
    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()

#feature extractor for depth input
#custom cnn that processes lowres depth map and confidence mask
class DepthEncoder(nn.Module):

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

        #init convolution
        self.stem = _conv_bn_relu(in_ch, channels[0], k=k)

        #downsampling layers
        self.down_layers = nn.ModuleList()
        self.down_layers.append(
            nn.Sequential(_conv_bn_relu(channels[0], channels[0], k=k), nn.MaxPool2d(2))
        )
        for i in range(1, levels):
            self.down_layers.append(
                nn.Sequential(_conv_bn_relu(channels[i - 1], channels[i], k=k), nn.MaxPool2d(2))
            )

    #returns depth features for stem and downsampled levels
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.stem(x)]
        for dl in self.down_layers:
            feats.append(dl(feats[-1]))
        return feats

    #retrieve channel dim at downsampling level
    def feat_ch(self, idx: int) -> int:
        return self.channels[0] if idx == 0 else self.channels[idx - 1]

#decoder block
#upsamples output of previous layers to match skips, concatenates everything, applies convolutions to fuse features
class FuseUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            _conv_bn_relu(in_ch, out_ch, k=3),
            _conv_bn_relu(out_ch, out_ch, k=3),
        )

    #align spatial dims with skip connections
    def forward(self, x: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
        target_size = skips[0].shape[-2:]
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        #concat along channel dim
        x = torch.cat([x] + list(skips), dim=1)
        return self.fuse(x)


_HEAD_CH = 16

#combines rgbencoder and depthencoder, fuses their outputs, and predicts highres depth map w residual
class RGBGuidedDepthUpsampler(nn.Module):

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
        self.residual_scale = residual_scale #scaling factor for predicted residual
        self._levels = depth_encoder_levels

        #init encoders
        self.rgb_enc = RGBEncoder(pretrained=pretrained_rgb)
        self.depth_enc = DepthEncoder(
            in_ch=2,
            levels=depth_encoder_levels,
            channels=depth_channels,
            filter_size=depth_filter_size,
        )

        levels = depth_encoder_levels
        rgb_chs = RGBEncoder.TAP_CHANNELS
        d_ch = self.depth_enc.feat_ch
        fdc = list(fusion_decoder_channels)[:levels]

        #fuses deepest depth and rgb features
        self.bot = _conv_bn_relu(d_ch(levels) + rgb_chs[levels - 1], fdc[0], k=3)

        #upsampling/fusion decoder path
        self.up_blocks = nn.ModuleList()
        for i in range(levels - 1):
            rgb_idx = levels - 2 - i
            d_idx = levels - 1 - i
            #total channels = input from previous decoder layer + rgb skip + depth skip
            in_ch = fdc[i] + rgb_chs[rgb_idx] + d_ch(d_idx)
            self.up_blocks.append(FuseUpBlock(in_ch, fdc[i + 1]))

        # final up to full resolution: fuse with depth stem only
        self.up0 = FuseUpBlock(fdc[levels - 1] + d_ch(0), _HEAD_CH)

        #final prediction head
        self.head = nn.Sequential(
            _conv_bn_relu(_HEAD_CH, _HEAD_CH, k=3),
            nn.Conv2d(_HEAD_CH, 1, 3, padding=1),
        )
        # zero init so training starts from the bicubic baseline
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _decode(
        self,
        rgb_feats: List[torch.Tensor],
        bicubic_norm: torch.Tensor,
        conf_hi: torch.Tensor,
    ) -> torch.Tensor:
        depth_in = torch.cat([bicubic_norm, conf_hi], dim=1)
        depth_feats = self.depth_enc(depth_in)
        levels = self._levels
        x = self.bot(torch.cat([depth_feats[levels], rgb_feats[levels - 1]], dim=1))
        for i, blk in enumerate(self.up_blocks):
            x = blk(x, rgb_feats[levels - 2 - i], depth_feats[levels - 1 - i])
        u0 = self.up0(x, depth_feats[0])
        return bicubic_norm + self.residual_scale * self.head(u0)

    #standard forward pass
    def forward(
        self,
        rgb: torch.Tensor,
        bicubic_norm: torch.Tensor,
        conf_hi: torch.Tensor,
    ) -> torch.Tensor:
        return self._decode(self.rgb_enc(rgb), bicubic_norm, conf_hi)

    #forward pass w frozen rgb encoder
    def forward_frozen(
        self,
        rgb: torch.Tensor,
        bicubic_norm: torch.Tensor,
        conf_hi: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            rgb_feats = self.rgb_enc(rgb)
        return self._decode(rgb_feats, bicubic_norm, conf_hi)

    #freeze rgb branch
    def freeze_rgb(self):
        self.rgb_enc.freeze()
    
    #unfreeze rgb branch
    def unfreeze_rgb(self):
        self.rgb_enc.unfreeze()

#count total model params and params requiring gradients
def count_parameters(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable
