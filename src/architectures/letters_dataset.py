import numpy as np
import torch
from torchvision import transforms


class LettersDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        # load your dataset (how every you want, this example has the dataset stored in a json file
        self.dataset = np.expand_dims(dataset, 1).astype('float32')

        # transforms_sequence = transforms.Compose( # Unnecessary for this case
        #     # transforms.ToPILImage(),
        #     # transforms.Grayscale(num_output_channels=1),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0.5], std=[0.1])
        # )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
