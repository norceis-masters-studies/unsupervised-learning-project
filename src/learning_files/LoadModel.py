# %%
import comet_ml

# %%
from io import BytesIO

import torch
import torch.nn.functional as F

# %%
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms

from AutoEncoder import Encoder, Decoder, Autoencoder

from tqdm import tqdm
import einops
import matplotlib.pyplot as plt

from comet_ml.api import Tag, Metric, Metadata

# %%
# Model
latent_width = 64
encoder = Encoder(latent_width)
decoder = Decoder(encoder.latent_width)
model = Autoencoder(encoder, decoder)

# %%
# Find the best experiment
api = comet_ml.API()
experiments = api.query(
    "paluczak",
    "UczenieNienadzorowane",
    (Metadata("file_name") == "AutoEncoder.py")
    & Tag("LOSS: l1_loss")
    & (Metric("train_loss") < 0.05),
)
print(experiments)
api_experiment = experiments[0]

# %%
# Load weights
print(api_experiment.get_name())
# print(api_experiment.get_asset_list()) # show this
model_asset = list(
    filter(
        lambda asset: asset["type"] == "model-element",
        api_experiment.get_asset_list(),
    )
)[0]
model_weights = torch.load(
    BytesIO(api_experiment.get_asset(model_asset["assetId"], return_type="binary"))
)

model.load_state_dict(model_weights)
model.cuda()
model.eval()

# %%
# Data
mnist_val = MNIST(
    "/datasets",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
val_loader = DataLoader(
    mnist_val,
    batch_size=8,
    num_workers=32,
    shuffle=False,
    pin_memory=True,
)

# %%
# Run model
loss_func = F.l1_loss
with torch.no_grad():
    losses = []

    for idx, batch in tqdm(enumerate(val_loader), desc="RUN"):

        images, labels = batch
        images = images.cuda()
        predictions, latents = model(images)
        loss = loss_func(predictions, images)

        losses.append(loss.item())

        if idx == 42:
            sample_images = einops.rearrange(
                [images, predictions],
                "source batch 1 height width -> (batch height) (source width)",
            ).cpu()

print(torch.mean(torch.tensor(losses)))
plt.figure(figsize=(4, 20))
plt.imshow(sample_images)
plt.show()

# %%
