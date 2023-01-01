import gc
import os

import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import src.utility.letter_operations as lo
from src.architectures import course_autoencoder, deep_clustering_convolution_autoencoder
from src.architectures.letters_dataset import LettersDataset


MODELS = {
    "course": course_autoencoder,
    "deep_clustering": deep_clustering_convolution_autoencoder
}


def extract_letters_from_pages(save: bool = False):

    directory_files_emnist = sorted(os.listdir("data/generated/streamlit/emnist/"), key=lambda f: int(''.join(filter(str.isdigit, f))))
    directory_files_kuzushiji = sorted(os.listdir("data/generated/streamlit/kuzushiji/"), key=lambda f: int(''.join(filter(str.isdigit, f))))
    page_list_emnist = []
    page_list_kuzushiji = []

    for file in directory_files_emnist:
        page_list_emnist.append(cv2.imread("data/generated/streamlit/emnist/" + file, cv2.IMREAD_GRAYSCALE))

    for file in directory_files_kuzushiji:
        page_list_kuzushiji.append(cv2.imread("data/generated/streamlit/kuzushiji/" + file, cv2.IMREAD_GRAYSCALE))

    number_of_pages = len(directory_files_emnist)

    sliced_page_list_emnist = page_list_emnist.copy()[:number_of_pages]
    sliced_page_list_kuzushiji = page_list_kuzushiji.copy()[:number_of_pages]

    letters_emnist = []

    for page in sliced_page_list_emnist:
        for i in range(0, page.shape[0], 32):
            for j in range(0, page.shape[1], 32):
                single_letter = page[i:i + 32, j:j + 32].copy()
                single_letter = lo.invert_pixels(single_letter) // 255
                letters_emnist.append(single_letter)

    letters_kuzushiji = []

    for page in sliced_page_list_kuzushiji:
        for i in range(0, page.shape[0], 32):
            for j in range(0, page.shape[1], 32):
                single_letter = page[i:i + 32, j:j + 32].copy()
                single_letter = lo.invert_pixels(single_letter) // 255
                letters_kuzushiji.append(single_letter)

    filtered_letters_emnist = []
    space_indices_emnist = []
    index = 0

    for letter in letters_emnist:
        if np.mean(letter) > 0.1:
            filtered_letters_emnist.append(letter)
        else:
            space_indices_emnist.append(index)
        index += 1

    filtered_letters_kuzushiji = []
    space_indices_kuzushiji = []
    index = 0

    for letter in letters_kuzushiji:
        if np.mean(letter) > 0.1:
            filtered_letters_kuzushiji.append(letter)
        else:
            space_indices_kuzushiji.append(index)
        index += 1

    if save:
        np.save('data/generated/streamlit/extracted_emnist', np.array(filtered_letters_emnist))
        np.save('data/generated/streamlit/space_indices_emnist', np.array(space_indices_emnist))

        np.save('data/generated/streamlit/extracted_kuzushiji', np.array(filtered_letters_kuzushiji))
        np.save('data/generated/streamlit/space_indices_kuzushiji', np.array(space_indices_kuzushiji))

    return np.array(filtered_letters_emnist), space_indices_emnist, \
        np.array(filtered_letters_kuzushiji), space_indices_kuzushiji


def encode_wrapper(encoder, dataset):
    device = torch.device("cpu") # "cuda:0" if torch.cuda.is_available() else

    encoder.to(device)

    val_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
    )

    preds = []

    with torch.no_grad() as nograd:
        for batch in tqdm(val_loader):
            images = batch.to(device)
            predictions = encoder(images)
            preds.append(predictions.cpu())

    preds_numpy = np.zeros((len(preds), 16))

    iterator = 0
    for tensor in preds:
        for block in range(tensor.shape[0]):
            preds_numpy[iterator] = tensor[block].numpy().copy()
            iterator += 1

    return preds_numpy


def encode_letters_emnist(model_name: str, emnist_letters, save: bool = False):

    if model_name == "course":
        emnist_model = MODELS[model_name].Autoencoder(MODELS[model_name].Encoder(16), MODELS[model_name].Decoder(16))
        emnist_model.load_state_dict(torch.load("data/models/1672100060_emnist_course_autoencoder.pth", map_location=torch.device('cpu')))
        emnist_encoder = emnist_model.encoder

    elif model_name == "deep_clustering":
        raise NotImplementedError

    mnist_dataset = LettersDataset(emnist_letters)

    encoded_emnist = encode_wrapper(emnist_encoder, mnist_dataset)

    if save:
        np.save("data/generated/streamlit/encoded_emnist", encoded_emnist)

    del emnist_model, emnist_encoder, mnist_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return encoded_emnist


def encode_letters_kuzushiji(model_name: str, kuzushiji_letters, save: bool = False):

    if model_name == "course":
        kuzushiji_model = MODELS[model_name].Autoencoder(MODELS[model_name].Encoder(16), MODELS[model_name].Decoder(16))
        kuzushiji_model.load_state_dict(torch.load("data/models/1672100367_kuzushiji_course_autoencoder.pth", map_location=torch.device('cpu')))
        kuzushiji_encoder = kuzushiji_model.encoder

    elif model_name == "deep_clustering":
        raise NotImplementedError

    kuzushiji_dataset = LettersDataset(kuzushiji_letters)

    encoded_kuzushiji = encode_wrapper(kuzushiji_encoder, kuzushiji_dataset)

    if save:
        np.save("data/generated/streamlit/encoded_kuzushiji", encoded_kuzushiji)

    del kuzushiji_encoder, kuzushiji_model, kuzushiji_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return encoded_kuzushiji