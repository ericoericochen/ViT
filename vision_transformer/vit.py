import torch
import torch.nn as nn
import math


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


class Transformer(nn.Module):
    pass


class TransformerEncoder(nn.Module):
    pass


class MultiHeadAttention(nn.Module):
    pass


class Attention(nn.Module):
    pass


class MLP(nn.Module):
    pass


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        layers: int,
        heads: int = 6,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_embedding = nn.Parameter(torch.zeros(dim))

        num_tokens = (image_size**2) // (patch_size**2) + 1  # add 1 for the CLS token
        positions = torch.arange(num_tokens)
        self.positional_embedding = timestep_embedding(positions, dim)

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

        return x
