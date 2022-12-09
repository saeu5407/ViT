'''모델 구현 참고: https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632'''

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# Embedding
class Embedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        # patch embedding
        # linear 모델 대신 conv layer를 사용하여 성능 향상
        self.projection = nn.Sequential(nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
                                        Rearrange('b e h w -> b (h w) e'),)
        # add CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # positional embedding
        '''
        왜 0~1사이 값을 쓰는가?
        왜 concat으로 붙이는것이 아니라 더할까?
        https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding#4d058603-db0f-4d62-bb49-d85ea6dcbfc6
        '''
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor):
        b, _, _, _ = x.shape
        x = self.projection(x) # patch embedding
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) # add CLS
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions # add positional embedding
        return x

# Residual Add
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# MLP
# nn.Sequential을 받으면 forward를 안해도 된다고 합니다.
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# EncoderBlock
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__()
        self.residual_1 = ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                                    MultiHeadAttention(emb_size, **kwargs),
                                                    nn.Dropout(drop_p)))
        self.residual_2 = ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                                    FeedForwardBlock(emb_size,
                                                                     expansion=forward_expansion,
                                                                     drop_p=forward_drop_p),
                                                    nn.Dropout(drop_p)))

    def forward(self, x: Tensor):
        residual_1 = self.residual_1(x)
        residual_2 = self.residual_2(residual_1)
        return residual_2
'''
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
'''

# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x
'''
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
'''

# ClassificationHead
class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__()
        self.classblock = nn.Sequential(Reduce('b n e -> b e', reduction='mean'),
                                        nn.LayerNorm(emb_size),
                                        nn.Linear(emb_size, n_classes))

    def forward(self, x: Tensor) -> Tensor:
        rtn = self.classblock(x)
        return rtn
'''
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))
'''

# ViT
class ViT(nn.Module):
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__()
        self.vit = nn.Sequential(Embedding(in_channels, patch_size, emb_size, img_size),
                                 TransformerEncoder(depth, emb_size=emb_size, **kwargs),
                                 ClassificationHead(emb_size, n_classes),)

    def forward(self, x: Tensor) -> Tensor:
        return self.vit(x)
'''
class ViT(nn.Sequential):
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            Embedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
'''

if __name__ == '__main__':

    summary(ViT(), (3, 224, 224), device='cpu')
