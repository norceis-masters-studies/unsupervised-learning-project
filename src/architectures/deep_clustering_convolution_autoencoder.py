# https://www.researchgate.net/publication/320658590_Deep_Clustering_with_Convolutional_Autoencoders

import torch
import einops
from torch import nn


class Encoder(nn.Module):
    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.model = nn.Sequential(
            Encoder.conv_block(1, 32),
            Encoder.conv_block(32, 64),
            Encoder.conv_block(64, 128),
            Encoder.conv_block(128, 256),

        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, self.latent_width),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = einops.rearrange(x, "b c w h -> b (c w h)")
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_size,
                out_size,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True
            ),
            # nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.fc = nn.Sequential(
            nn.Linear(self.latent_width, 256 * 2 * 2),
        )
        self.model = nn.Sequential(
            Decoder.conv_block(256, 128),
            Decoder.conv_block(128, 64),
            Decoder.conv_block(64, 32),
            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True
            ),
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = einops.rearrange(x, "b (c w h) -> b c w h", c=256, w=2, h=2)
        x = self.model(x)

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
