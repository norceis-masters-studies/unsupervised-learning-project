import torch
import einops
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
        )

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.model = nn.Sequential(
            nn.Linear(32 * 32, 64 * 64),
            nn.LeakyReLU(),
            nn.Linear(64 * 64, 32 * 32),
            nn.LeakyReLU(),
            nn.Linear(32 * 32, 16 * 16),
            nn.LeakyReLU(),
            nn.Linear(16 * 16, 8 * 8),
            nn.LeakyReLU(),
            nn.Linear(8 * 8, self.latent_width)
        )

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.model = nn.Sequential(
            nn.Linear(self.latent_width, 8 * 8),
            nn.LeakyReLU(),
            nn.Linear(8 * 8, 16 * 16),
            nn.LeakyReLU(),
            nn.Linear(16 * 16, 32 * 32),
            nn.LeakyReLU(),
            nn.Linear(32 * 32, 64 * 64),
            nn.LeakyReLU(),
            nn.Linear(64 * 64, 32*32),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = einops.rearrange(x, "b (c w h) -> b c w h", c=1, w=32, h=32)

        return x


class Autoencoder(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor):
        latent = self.encoder(images)
        decoded = self.decoder(latent)

        return decoded, latent
