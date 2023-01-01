from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
import emnist
import numpy as np


def load_emnist_bymerge_mapping() -> Dict:
    """
    Load mapping of EMNIST dataset bymerge. Mapped by label:character.
    :return: Dict
    """
    character_mapping = dict()

    emnist_csv = pd.read_csv("data/emnist/emnist-bymerge-mapping.txt", delimiter=" ", index_col=0,
                             header=None)

    emnist_ndarray = emnist_csv.to_numpy()

    for idx in range(len(emnist_ndarray)):
        character_mapping[idx] = chr(int(emnist_ndarray[idx]))

    return character_mapping


def load_kuzushiji_mapping() -> Dict:
    """
    Load mapping of KUZUSHIJI-49 dataset. Mapped by label:character.
    :return: Dict
    """
    character_mapping = dict()

    kuzushiji_csv = pd.read_csv("data/kuzushiji-49/k49_classmap.csv", index_col=0)

    kuzushiji_ndarray = kuzushiji_csv.to_numpy()

    for idx in range(len(kuzushiji_ndarray)):
        character_mapping[idx] = kuzushiji_ndarray[idx][1]

    return character_mapping


def load_emnist_bymerge() -> Tuple[Dict, Dict]:
    """
    Load EMNIST dataset bymerge.
    Return tuple containing dictinary grouped by label and dictionary of mapping label:character.
    :return: Tuple[Dict, Dict]
    """
    character_dict = dict()

    character_mapping = load_emnist_bymerge_mapping()

    images, labels = emnist.extract_training_samples("bymerge")

    balanced_32 = np.pad(images, ((0, 0), (2, 2), (2, 2)), mode='constant')

    for index in list(character_mapping.keys()):
        character_dict[index] = balanced_32[np.where(labels == index)]
        # @TODO add some logging
        # print(character_mapping[index], " loaded as number " + str(index))

    return character_dict, character_mapping


def load_kuzushiji():
    """
    Load KUZUSHIJI-49 dataset.
    Return tuple containing dictinary grouped by label and dictionary of mapping label:character.
    :return: Tuple[Dict, Dict]
    """
    character_dict = dict()

    character_mapping = load_kuzushiji_mapping()

    images = np.load("data/kuzushiji-49/k49-train-imgs.npz")['arr_0']
    labels = np.load("data/kuzushiji-49/k49-train-labels.npz")['arr_0']

    kuzushiji_32 = np.pad(images, ((0, 0), (2, 2), (2, 2)), mode='constant')

    for index in list(character_mapping.keys()):
        character_dict[index] = kuzushiji_32[np.where(labels == index)]
        # @TODO add some logging
        # print(character_mapping[index], " loaded as number " + str(index))

    return character_dict, character_mapping


if __name__ == "__main__":
    emnist, map = load_emnist_bymerge()
    print(emnist[46][0].shape)
    plt.imshow(emnist[46][0])
    plt.show()

    # kuzu, map = load_kuzushiji()
    # # print(emnist)
    # plt.imshow(kuzu[46][32])
    # plt.show()
