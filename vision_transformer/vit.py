import torch
import torch.nn as nn
import math
from einops import rearrange


# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/nn.py
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Params:
        - q: tensor of shape (B, L, d_q)
        - k: tensor of shape (B, L, d_k)
        - v: tensor of shape (B, L, d_v)

    All query and key should have the same dimension d_q=d_k.
    """
    d_k = k.shape[-1]

    k_t = k.transpose(-2, -1)  # (B, D, L)
    attention = nn.functional.softmax((q @ k_t) / (d_k**0.5), dim=-1) @ v  # (B, L, D)

    return attention


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 128,
    ):
        super().__init__()

        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim

        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    dim=dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for encoder in self.encoders:
            x = encoder(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, mlp_dim: int = 128
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim

        self.mha = MultiHeadAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_dim=mlp_dim)

    def forward(self, x: torch.Tensor):
        """
        Params:
            - x: (B, L, D)
        """
        h = x

        x = self.mha(self.layer_norm1(x)) + h
        x = self.mlp(self.layer_norm1(x)) + h

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        assert (
            D == self.dim
        ), f"Input dimension {D} does not match expected dimension {self.dim}"

        qkv = self.to_qkv(x)  # (B, L, dim) -> (B, L, 3 * heads * dim_head)
        qkv = qkv.reshape(
            (B, L, 3, self.heads, self.dim_head)
        )  # (B, L, 3 * heads * dim_head) -> (B, L, 2, heads, dim_head)

        qkv = rearrange(
            qkv, "b l v h d -> (b h) v l d"
        )  # (B, L, 3, heads, dim_head) -> (B, 3, heads, L, dim_head)

        # split into q k v vectors
        q, k, v = qkv.unbind(dim=1)  # (B x heads, L, dim_head)

        x = scaled_dot_product_attention(q, k, v)  # (B x heads, L, dim_head)
        x = rearrange(
            x, "(b h) l d -> b l (d h)", b=B, h=self.heads
        )  # (B x heads, L, dim_head) -> (B, L dim_head x heads)

        # linear projection to dim
        x = self.out(x)

        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        layers: int,
        channels: int = 3,
        heads: int = 6,
        mlp_dim: int = 128,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.layers = layers
        self.channels = channels
        self.heads = heads
        self.mlp_dim = mlp_dim

        self.patch_embedding = nn.Conv2d(
            in_channels=channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_embedding = nn.Parameter(torch.zeros(dim))

        num_tokens = (image_size**2) // (patch_size**2) + 1  # add 1 for the CLS token
        positions = torch.arange(num_tokens)
        self.positional_embedding = nn.Parameter(
            timestep_embedding(positions, dim), requires_grad=False
        )

        self.transformer = Transformer(
            dim=dim, layers=layers, heads=heads, mlp_dim=mlp_dim
        )

        self.layernorm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, num_classes),
        )

    def to_patch_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            - x: tensor of shape (B, C, H, W)
        """
        B = x.shape[0]

        patch_embeddings = self.patch_embedding(
            x
        )  # (B, C, H, W) -> (B, D, H // P, W // P)

        # flatten patch embeddings
        patch_embeddings = patch_embeddings.flatten(start_dim=2, end_dim=3).permute(
            (0, 2, 1)
        )  # (B, D, H // P, W // P) -> (B, HW//P^2, D)

        # add [CLS] token
        cls_token = self.cls_embedding.repeat(B, 1, 1)  # (B, 1, D)
        patch_embeddings = torch.cat(
            [cls_token, patch_embeddings], dim=1
        )  # (B, HW//P^2 + 1, D)

        return patch_embeddings

    def forward(self, x: torch.Tensor):
        x = self.to_patch_embedding(x)

        # add positional embeddings to patch embeddings
        x = x + self.positional_embedding

        # transformer
        x = self.transformer(x)

        # get the [CLS] token
        cls = x[:, 1, :]
        cls = self.layernorm(cls)

        # get classification
        x = self.mlp(cls)

        return x
