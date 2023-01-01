# %%
import comet_ml

# %%
import torch
from torch import nn
from torch.nn import functional as F

import einops

# %%
class Encoder(nn.Module):
    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
        )

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.model = nn.Sequential(
            Encoder.conv_block(1, 8),
            Encoder.conv_block(8, 8),
            Encoder.conv_block(8, 16),
            Encoder.conv_block(16, 16),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 2 * 2, self.latent_width),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        # x = torch.flatten(x, start_dim=1)
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
            ),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
        )

    def __init__(self, latent_width: int = 64, **kwargs) -> None:
        super().__init__()
        self.latent_width = latent_width

        self.fc = nn.Sequential(
            nn.Linear(self.latent_width, 16 * 2 * 2),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            Decoder.conv_block(16, 16),
            Decoder.conv_block(16, 8),
            Decoder.conv_block(8, 8),
            Decoder.conv_block(8, 8),
            nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),  # end result is an image, it must be in [0, 1] range!
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        # x = torch.reshape(x, (-1, 16, 2, 2))
        x = einops.rearrange(x, "b (c w h) -> b c w h", c=16, w=2, h=2)
        x = self.model(x)
        return x


# %%
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


# %%
import os

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms

from torchinfo import summary

from tqdm import tqdm

if __name__ == "__main__":
    # data
    mnist_train = MNIST(
        "/datasets",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    mnist_val = MNIST(
        "/datasets",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(
        mnist_train,
        batch_size=32,
        num_workers=32,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        mnist_val,
        batch_size=32,
        num_workers=32,
        shuffle=False,
        pin_memory=True,
    )

    # model
    latent_width = 64
    encoder = Encoder(latent_width)
    decoder = Decoder(encoder.latent_width)
    model = Autoencoder(encoder, decoder)
    model.cuda()

    # logging
    comet_experiment = comet_ml.Experiment(project_name="UczenieNienadzorowane")
    comet_experiment.log_code(folder="UN")
    comet_experiment.log_parameters(
        {
            "batch_size": train_loader.batch_size,
            "train_size": len(mnist_train),
            "val_size": len(mnist_val),
        }
    )

    summ = summary(model, (1, 1, 32, 32), device="cuda", depth=5)
    comet_experiment.set_model_graph(f"{model.__repr__()}\n{summ}")

    # pure PyTorch loop
    num_epochs = 5
    # loss_func = F.mse_loss
    loss_func = F.l1_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    comet_experiment.log_parameter("num_epochs", num_epochs)
    comet_experiment.log_parameter("loss_func", loss_func.__name__)

    comet_experiment.add_tag(f"LOSS: {loss_func.__name__}")

    for epoch in range(num_epochs):
        comet_experiment.set_epoch(epoch)

        model.eval()
        with comet_experiment.validate() as validat, torch.no_grad() as nograd:
            for idx, batch in tqdm(enumerate(val_loader), desc=f"VAL_{epoch}"):
                comet_experiment.set_step(idx + epoch * len(val_loader))

                images, labels = batch
                images = images.cuda()
                predictions, latents = model(images)
                loss = loss_func(predictions, images)

                comet_experiment.log_metric("loss", loss.item())
                images = einops.rearrange(
                    [images, predictions],
                    "source batch 1 height width -> batch height (source width)",
                ).cpu()
                comet_experiment.log_image(images[0], f"images_{idx}", step=epoch)

        model.train()
        with comet_experiment.train() as train:
            for idx, batch in tqdm(enumerate(train_loader), desc=f"TRAIN_{epoch}"):
                comet_experiment.set_step(idx + epoch * len(train_loader))

                # look at: model.encoder.parameters().__next__()
                # look at: model.encoder.parameters().__next__().grad
                # look at: latents.shape

                optimizer.zero_grad()  # MUST be called on every batch
                images, labels = batch
                images = images.cuda()
                predictions, latents = model(images)
                loss = loss_func(predictions, images)
                loss.backward()
                optimizer.step()

                comet_experiment.log_metric("loss", loss.item())
                comet_experiment.log_histogram_3d(latents.detach().cpu(), "latents")

    model.eval()
    with comet_experiment.test() as test, torch.no_grad() as nograd:
        for idx, batch in tqdm(enumerate(val_loader), desc=f"TEST_{num_epochs}"):
            comet_experiment.set_step(idx + num_epochs * len(val_loader))

            images, labels = batch
            images = images.cuda()
            predictions, latents = model(images)
            loss = loss_func(predictions, images)

            comet_experiment.log_metric("loss", loss.item())
            images = einops.rearrange(
                [images, predictions],
                "source batch 1 height width -> (batch height) (source width)",
            ).cpu()
            comet_experiment.log_image(images, f"images_{idx}", step=num_epochs)

    # Log model
    model_name = f"{type(model).__name__}_{comet_experiment.get_key()}"
    torch.save(model.state_dict(), f"{model_name}.pth")
    comet_experiment.log_model(model_name, f"./{model_name}.pth")
    os.remove(f"./{model_name}.pth")
