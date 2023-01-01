from pathlib import Path

import cv2
import os
import numpy as np
import src.utility.letter_operations as lo

def load_emnist_pages(number_of_pages: int = 0, trial: str = 'all'):
    """
    Load pages with EMNIST symbols.
    Return NumPy array containing extracted EMNIST symbols without spaces.
    :return: NumPy array
    """

    path_emnist = 'data/generated/emnist/' + trial + '/'

    directory_files_emnist = sorted(os.listdir(path_emnist), key=lambda f: int(''.join(filter(str.isdigit, f))))
    page_list_emnist = []

    for file in directory_files_emnist:
        page_list_emnist.append(cv2.imread(path_emnist + file, cv2.IMREAD_GRAYSCALE))

    if number_of_pages == 0:
        number_of_pages = len(directory_files_emnist)

    sliced_page_list_emnist = page_list_emnist.copy()[:number_of_pages]

    letters_emnist = []

    for page in sliced_page_list_emnist:
        for i in range(0, page.shape[0], 32):
            for j in range(0, page.shape[1], 32):
                single_letter = page[i:i + 32, j:j + 32].copy()
                single_letter = lo.invert_pixels(single_letter) // 255
                letters_emnist.append(single_letter)

    filtered_letters_emnist = np.array([x for x in letters_emnist if np.mean(x) > 0.19])

    return filtered_letters_emnist


def load_kuzushiji_pages(number_of_pages: int = 0, trial: str = 'all'):
    """
    Load pages with KUZUSHIJI-49 symbols.
    Return NumPy array containing extracted KUZUSHIJI-49 symbols without spaces.
    :return: NumPy array
    """

    path_kuzushiji = 'data/generated/kuzushiji/' + trial + '/'

    directory_files_kuzushiji = sorted(os.listdir(path_kuzushiji), key=lambda f: int(''.join(filter(str.isdigit, f))))
    page_list_kuzushiji = []

    for file in directory_files_kuzushiji:
        page_list_kuzushiji.append(cv2.imread(path_kuzushiji + file, cv2.IMREAD_GRAYSCALE))

    if number_of_pages == 0:
        number_of_pages = len(directory_files_kuzushiji)

    sliced_page_list_kuzushiji = page_list_kuzushiji.copy()[:number_of_pages]

    letters_kuzushiji = []

    for page in sliced_page_list_kuzushiji:
        for i in range(0, page.shape[0], 32):
            for j in range(0, page.shape[1], 32):
                single_letter = page[i:i + 32, j:j + 32].copy()
                single_letter = lo.invert_pixels(single_letter) // 255
                letters_kuzushiji.append(single_letter)

    filtered_letters_kuzushiji = np.array([x for x in letters_kuzushiji if np.mean(x) > 0.19])

    return filtered_letters_kuzushiji

def load_emnist_pages_with_spaces(number_of_pages: int = 0, trial: str = 'all'):
    """
    Load pages with EMNIST symbols.
    Return NumPy array containing extracted EMNIST symbols without spaces.
    :return: NumPy array
    """

    path_emnist = 'data/generated/emnist/' + trial + '/'

    directory_files_emnist = sorted(os.listdir(path_emnist), key=lambda f: int(''.join(filter(str.isdigit, f))))
    page_list_emnist = []

    for file in directory_files_emnist:
        page_list_emnist.append(cv2.imread(path_emnist + file, cv2.IMREAD_GRAYSCALE))

    if number_of_pages == 0:
        number_of_pages = len(directory_files_emnist)

    sliced_page_list_emnist = page_list_emnist.copy()[:number_of_pages]

    letters_emnist = []

    for page in sliced_page_list_emnist:
        for i in range(0, page.shape[0], 32):
            for j in range(0, page.shape[1], 32):
                single_letter = page[i:i + 32, j:j + 32].copy()
                single_letter = lo.invert_pixels(single_letter) // 255
                letters_emnist.append(single_letter)

    filtered_letters_emnist = []
    space_indices_emnist = []
    index = 0

    for letter in letters_emnist:
        if np.mean(letter) > 0.19:
            filtered_letters_emnist.append(letter)
        else:
            space_indices_emnist.append(index)
        index += 1

    return np.array(filtered_letters_emnist), space_indices_emnist


def load_kuzushiji_pages_with_spaces(number_of_pages: int = 0, trial: str = 'all'):
    """
    Load pages with KUZUSHIJI-49 symbols.
    Return NumPy array containing extracted KUZUSHIJI-49 symbols without spaces.
    :return: NumPy array
    """

    path_kuzushiji = 'data/generated/kuzushiji/' + trial + '/'

    directory_files_kuzushiji = sorted(os.listdir(path_kuzushiji), key=lambda f: int(''.join(filter(str.isdigit, f))))
    page_list_kuzushiji = []

    for file in directory_files_kuzushiji:
        page_list_kuzushiji.append(cv2.imread(path_kuzushiji + file, cv2.IMREAD_GRAYSCALE))

    if number_of_pages == 0:
        number_of_pages = len(directory_files_kuzushiji)

    sliced_page_list_kuzushiji = page_list_kuzushiji.copy()[:number_of_pages]

    letters_kuzushiji = []

    for page in sliced_page_list_kuzushiji:
        for i in range(0, page.shape[0], 32):
            for j in range(0, page.shape[1], 32):
                single_letter = page[i:i + 32, j:j + 32].copy()
                single_letter = lo.invert_pixels(single_letter) // 255
                letters_kuzushiji.append(single_letter)

    filtered_letters_kuzushiji = []
    space_indices_kuzushiji = []
    index = 0

    for letter in letters_kuzushiji:
        if np.mean(letter) > 0.19:
            filtered_letters_kuzushiji.append(letter)
        else:
            space_indices_kuzushiji.append(index)
        index += 1

    return np.array(filtered_letters_kuzushiji), space_indices_kuzushiji