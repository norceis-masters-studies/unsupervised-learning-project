import streamlit as st
from streamlit_option_menu import option_menu

import re
from pathlib import Path
from typing import Tuple, Dict
import time
import copy
import pandas as pd
import emnist

from typing import Union

from PIL import Image
import cv2
import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from umap import UMAP
from yellowbrick.cluster import KElbowVisualizer
import warnings
import os

# print(os.getcwd())
from src.utility.streamlit_functions import extract_letters_from_pages, encode_letters_emnist, \
    encode_letters_kuzushiji

from src.utility.dataset_loader import load_emnist_bymerge, load_kuzushiji
from src.utility.letter_operations import rotate_cv2, thresholding, invert_pixels, add_noise, stretch_cv2
from src.utility.symbols_loader import load_emnist_pages, load_kuzushiji_pages, load_emnist_pages_with_spaces, \
    load_kuzushiji_pages_with_spaces


ENCODED_EMNIST_PATH = "data/generated/streamlit/emnist_preds.npz"
ENCODED_KUZUSHIJI_PATH = "data/generated/streamlit/kuzushiji_preds.npz"
DECODED_EMNIST = "data/generated/streamlit/emnist_decoded.npz"


with st.sidebar:
    selection = option_menu(None, ["Generating pages", 'Extracting letters from pages',
                                   'Generating encoded representations of data',
                                   'Cluster visualisation',
                                   'Generating translated pages'],
                            icons=['newspaper', 'box-arrow-right', 'hypnotize', 'layers-half', 'journal'],
                            default_index=0)  # https://icons.getbootstrap.com/

if selection == 'Generating pages':

    mapping_type = st.selectbox('Choose the type of printing mappings:', ['All mappings', 'Few random mappings'])

    col1_1, col2_1, col3_1 = st.columns(3)
    if col2_1.button('Engage data generation'):
        with st.spinner('Loading data in progress'):
            NUMBER_OF_CHARACTERS = 5
            NOISE = 0.8

            FILE_PATH = Path().parent.as_posix() + "/"

            # np.random.seed(1)

            plt.style.use('grayscale')

            # sheet parameters
            columns = 80
            rows = 114
            cell = 32
            max_chars = columns * rows

            # st.subheader('Loading data in progress')
            kuzushiji_letters, kuzushiji_mapping = load_kuzushiji()
            del kuzushiji_mapping[48]  # deleting iterative character
            emnist_letters, emnist_mapping = load_emnist_bymerge()
            emnist_mapping_inverted = {v: k for k, v in emnist_mapping.items()}

            with open('data/winnie_the_pooh/pg67098.txt', 'r') as file:
                data = file.read()
            pattern = re.compile('^[a-zA-Z0-9 ]*$')
            result = "".join([s for s in data if pattern.match(s)])
            result = list(result)
            # time.sleep(1)
        st.success('Loading data done!')

        empty_sheet = np.zeros(shape=(rows * cell, columns * cell))
        temporary_list = list(kuzushiji_mapping.keys())
        np.random.shuffle(temporary_list)

        emnist_to_kuzushiji_mapping = dict()
        for element in range(len(emnist_mapping.keys())):
            emnist_to_kuzushiji_mapping[emnist_mapping[element]] = temporary_list[element]

        missing_letters = ['u', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 'c', 'y', 'v', 's', 'w', 'x', 'z']

        for ml in missing_letters:
            emnist_mapping_inverted[ml] = emnist_mapping_inverted[ml.upper()]
            emnist_to_kuzushiji_mapping[ml] = emnist_to_kuzushiji_mapping[ml.upper()]

        # time.sleep(1)
        st.success('Mapping done')

        col1_3, col2_3, col3_3, col4_3, col5_3 = st.columns(5)
        with col3_3:
            if mapping_type == 'All mappings':
                for element in range(len(emnist_mapping.keys())):
                    st.subheader(f'{emnist_mapping[element]} ➜ {kuzushiji_mapping[temporary_list[element]]}')
                    time.sleep(0.1)

            elif mapping_type == 'Few random mappings':
                for _ in range(5):
                    choice = np.random.choice(list(emnist_mapping.keys()))
                    st.subheader(f'{emnist_mapping[choice]} ➜ {kuzushiji_mapping[temporary_list[choice]]}')
                    time.sleep(0.1)

        with st.spinner('Generating kuzushiji pages'):
            l_idx = 0
            sheets_kuzushiji = list()

            bar_1 = st.progress(0)
            while l_idx < len(result):
                bar_1.progress(l_idx / (len(result) - 1))
                kuzushiji_sheet = empty_sheet.copy()
                for i in range(0, kuzushiji_sheet.shape[0], 32):
                    if l_idx >= len(result):
                        break
                    for j in range(0, kuzushiji_sheet.shape[1], 32):
                        if l_idx >= len(result):
                            break
                        if result[l_idx] == '\n':
                            l_idx += 1
                            break
                        elif result[l_idx] == ' ':
                            kuzushiji_sheet[i:i + 32, j:j + 32] = (np.zeros(shape=(32, 32)))
                        else:
                            random_choice = np.random.randint(
                                NUMBER_OF_CHARACTERS)  # np.random.choice(kuzushiji_letters[emnist_to_kuzushiji_mapping[result[l_idx]]].shape[0], 1)[0]
                            while random_choice >= kuzushiji_letters[emnist_to_kuzushiji_mapping[result[l_idx]]].shape[
                                0]:
                                random_choice = np.random.randint(NUMBER_OF_CHARACTERS)
                            plain_character = kuzushiji_letters[emnist_to_kuzushiji_mapping[result[l_idx]]][
                                random_choice]
                            if np.random.random() < 0.3:
                                plain_character = stretch_cv2(plain_character,
                                                              np.round(np.random.uniform(0.85, 1.15), 2),
                                                              np.random.randint(0, 2))
                            if np.random.random() < 0.3:
                                plain_character = rotate_cv2(plain_character, np.round(np.random.uniform(-30, 30)))
                            kuzushiji_sheet[i:i + 32, j:j + 32] = plain_character
                        l_idx += 1
                kuzushiji_sheet = thresholding(kuzushiji_sheet, 128)
                kuzushiji_sheet = add_noise(kuzushiji_sheet, NOISE)
                kuzushiji_sheet = invert_pixels(kuzushiji_sheet)
                sheets_kuzushiji.append(kuzushiji_sheet)

            bar_1.progress(100)
            path = 'data/generated/streamlit/kuzushiji/'

            for idx in range(len(sheets_kuzushiji)):
                cv2.imwrite(path + "/kuzushiji_" + str(idx) + ".png", sheets_kuzushiji[idx])
        st.success('Successfully generated kuzushiji pages')

        with st.spinner('Generating emnist pages'):
            l_idx = 0
            sheets_emnist = list()
            bar_2 = st.progress(0)

            while l_idx < len(result):
                bar_2.progress(l_idx / (len(result) - 1))
                emnist_sheet = empty_sheet.copy()
                for i in range(0, emnist_sheet.shape[0], 32):
                    if l_idx >= len(result):
                        break
                    for j in range(0, emnist_sheet.shape[1], 32):
                        if l_idx >= len(result):
                            break
                        if result[l_idx] == '\n':
                            l_idx += 1
                            break
                        elif result[l_idx] == ' ':
                            emnist_sheet[i:i + 32, j:j + 32] = (np.zeros(shape=(32, 32)))
                        else:
                            random_choice = np.random.randint(
                                NUMBER_OF_CHARACTERS)  # np.random.choice(emnist_letters[emnist_mapping_inverted[result[l_idx]]].shape[0], 1)[0]
                            while random_choice >= emnist_letters[emnist_mapping_inverted[result[l_idx]]].shape[0]:
                                random_choice = np.random.randint(NUMBER_OF_CHARACTERS)
                            plain_character = emnist_letters[emnist_mapping_inverted[result[l_idx]]][random_choice]
                            if np.random.random() < 0.3:
                                plain_character = stretch_cv2(plain_character,
                                                              np.round(np.random.uniform(0.85, 1.15), 2),
                                                              np.random.randint(0, 2))
                            if np.random.random() < 0.3:
                                plain_character = rotate_cv2(plain_character, np.round(np.random.uniform(-30, 30)))
                            emnist_sheet[i:i + 32, j:j + 32] = plain_character
                        l_idx += 1

                emnist_sheet = thresholding(emnist_sheet, 128)
                emnist_sheet = add_noise(emnist_sheet, NOISE)
                emnist_sheet = invert_pixels(emnist_sheet)
                sheets_emnist.append(emnist_sheet)

            bar_2.progress(100)
            path = 'data/generated/streamlit/emnist/'

            for idx in range(len(sheets_emnist)):
                cv2.imwrite(path + "/emnist_" + str(idx) + ".png", sheets_emnist[idx])

        st.success('Successfully generated emnist pages')

        col1_2, col2_2, col3_2 = st.columns(3)
        col2_2.text('Comparing random pages')

        plt.rcParams["figure.figsize"] = (8, 8)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        rng = np.random.randint(0, 35)
        path = 'data/generated/streamlit/'

        ax1.imshow(sheets_emnist[rng])
        ax1.axis('off')

        ax2.imshow(sheets_kuzushiji[rng])
        ax2.axis('off')

        st.pyplot(fig)
        st.balloons()

elif selection == 'Extracting letters from pages':

    # selector not really needed?
    # available_pages = \
    #     sorted(os.listdir('data/generated/streamlit/emnist/'), key=lambda x: int(x.split('_')[1].split('.')[0])) + \
    #     sorted(os.listdir('data/generated/streamlit/kuzushiji/'), key=lambda x: int(x.split('_')[1].split('.')[0]))
    #
    # selected_page = st.selectbox('Select a page', available_pages)

    st.title('Extract letters from generated pages')

    EXTRACTED_LETTERS = os.path.exists('data/generated/streamlit/extracted_emnist.npy') and \
                        os.path.exists('data/generated/streamlit/extracted_kuzushiji.npy')

    info_widget = st.empty()

    if EXTRACTED_LETTERS:
        info_widget = st.info('Letters are already extracted, you can continue to the next step or extract them again')

    _, col, _ = st.columns(3)
    if col.button('Extract letters from pages'):
        info_widget.empty()
        with st.spinner('Extracting letters from emnist pages'):
            emnist_extracted, emnist_indexes, kuzushiji_extracted, kuzushiji_indexes = \
                extract_letters_from_pages(save=True)
            st.success('Successfully extracted letters from emnist pages')
        with st.spinner('Showing random extracted letters from both datasets'):
            plt.rcParams["figure.figsize"] = (8, 8)
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

            # sizes might be different
            rng = np.random.randint(0, np.min([len(emnist_extracted), len(kuzushiji_extracted)]))

            ax1.imshow(emnist_extracted[rng])
            ax1.axis('off')
            ax1.title.set_text('EMNIST')

            ax2.imshow(kuzushiji_extracted[rng])
            ax2.axis('off')
            ax2.title.set_text('KUZUSHIJI')

            st.header('Random extracted letters from both datasets')

            st.pyplot(fig)
            info_widget = st.info('Letters are already extracted, you can continue to the next step or extract them again')
        st.balloons()

elif selection == 'Generating encoded representations of data':

    st.title('Generating encoded representations of data')




    _, col, _ = st.columns(3)
    if col.button('Generate encoded representations'):

        with st.spinner('Generating encoded emnist representations'):
            encoded_emnist = np.load(ENCODED_EMNIST_PATH)['arr_0']

            st.header('Encoded emnist representations')
            emnist_df = pd.DataFrame(encoded_emnist)

            st.success('Successfully generated encoded representations for emnist letters')
            st.table(emnist_df.head(10))

        with st.spinner('Generating encoded kuzushiji representations'):
            encoded_kuzushiji = np.load(ENCODED_KUZUSHIJI_PATH)['arr_0']

            st.header('Encoded kuzushiji representations')
            kuzushiji_df = pd.DataFrame(encoded_kuzushiji)

            st.success('Successfully generated encoded representations for kuzushiji letters')
            st.table(kuzushiji_df.head(10))

        st.balloons()
    # st.balloons()

elif selection == 'Cluster visualisation':

    dimensionality_reductions = st.multiselect('Choose the type of dimensionality reduction to run:',
                                               ['UMAP', 'TSNE'])  # , 'MDS'])
    clustering_algorithms = st.multiselect('Choose the type of clustering method to run:',
                                           ['KMeans', 'Spectral clustering', 'DBSCAN', 'Gaussian Mixture'])
    min_cluster, max_cluster = st.slider('Select minimum and maximum number of clusters to search through:',
                                         0, 100, (30, 40))
    number_of_samples = st.number_input('Choose number of samples from data', 0, 2000000, 2000000)
    # full_GM = False
    # if 'Gaussian Mixture' in clustering_algorithms:
    #     full_GM = st.checkbox('Do a full Gaussian mixture search')

    if 'DBSCAN' in clustering_algorithms:
        eps = st.slider('Choose number of eps for DBSCAN', 0.0, 2.0, 0.1)
        min_samples = st.slider('Choose number of minimum samples for DBSCAN', 0, 100, 5)

    col1_1, col2_1, col3_1, col4_1, col5_1 = st.columns(5)
    if col3_1.button('Engage clustering'):

        warnings.filterwarnings("ignore")

        BATCH_SIZE = 1000
        # path = "data/encoded_data/1-2-3-4-5/trial_4"

        with st.spinner('Loading data in progress'):
            emnist_pred = np.load(ENCODED_EMNIST_PATH)['arr_0']
            kuzushiji_pred = np.load(ENCODED_KUZUSHIJI_PATH)['arr_0']
        st.success('Loading data done!')

        emnist_pred = emnist_pred[np.random.choice(
            range(len(emnist_pred)), np.min([len(emnist_pred), number_of_samples]), replace=False)]
        kuzushiji_pred = kuzushiji_pred[np.random.choice(
            range(len(kuzushiji_pred)), np.min([len(kuzushiji_pred), number_of_samples]), replace=False)]


        def plot_representations(emnist, kuzushiji, markdown):
            plt.rcParams["figure.figsize"] = (14, 8)
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

            st.markdown(markdown, unsafe_allow_html=True)
            ax1.scatter(emnist[:, 0], emnist[:, 1], s=5, alpha=0.75)
            ax1.axis('off')
            ax1.title.set_text('Emnist')

            ax2.scatter(kuzushiji[:, 0], kuzushiji[:, 1], s=5, alpha=0.75)
            ax2.axis('off')
            ax2.title.set_text('Kuzushiji')

            st.pyplot(fig)


        if 'UMAP' in dimensionality_reductions:
            with st.spinner('Calculating UMAP representations'):
                reducer_emnist = UMAP(n_components=2)
                emnist_umap = reducer_emnist.fit_transform(emnist_pred)

                reducer_kuzushiji = UMAP(n_components=2)
                kuzushiji_umap = reducer_kuzushiji.fit_transform(kuzushiji_pred)

            st.success('Successfully calculated UMAP representations!')
            plot_representations(emnist_umap, kuzushiji_umap,
                                 "<h5 style='text-align: center; color: white;'>UMAP representations</h5>")

        if 'TSNE' in dimensionality_reductions:
            with st.spinner('Calculating TSNE representations'):
                reducer_emnist = TSNE(n_components=2)
                emnist_tsne = reducer_emnist.fit_transform(emnist_pred)

                reducer_kuzushiji = TSNE(n_components=2)
                kuzushiji_tsne = reducer_kuzushiji.fit_transform(kuzushiji_pred)

            st.success('Successfully calculated TSNE representations!')
            plot_representations(emnist_tsne, kuzushiji_tsne,
                                 "<h5 style='text-align: center; color: white;'>TSNE representations</h5>")

        if 'MDS' in dimensionality_reductions:
            with st.spinner('Calculating MDS representations'):
                emnist_mds = []
                reducer_emnist = MDS(n_components=2)
                for batch in range(math.ceil(len(emnist_pred) / BATCH_SIZE)):  # MDS doesn's work very well
                    emnist_mds.append(
                        reducer_emnist.fit_transform(emnist_pred[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]))
                emnist_mds = np.asarray(emnist_mds)

                kuzushiji_mds = []
                reducer_kuzushiji = MDS(n_components=2)
                for batch in range(math.ceil(len(kuzushiji_pred) / BATCH_SIZE)):
                    kuzushiji_mds.append(
                        reducer_kuzushiji.fit_transform(kuzushiji_pred[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]))
                kuzushiji_mds = np.asarray(kuzushiji_mds)

            st.success('Successfully calculated MDS representations! (Took a while huh?)')
            plot_representations(emnist_mds, kuzushiji_mds,
                                 "<h5 style='text-align: center; color: white;'>UMAP representations</h5>")

        emnist_scaled = StandardScaler().fit_transform(emnist_pred)
        kuzushiji_scaled = StandardScaler().fit_transform(kuzushiji_pred)


        def plot_Spectral(representation_emnist, emnist_clusters, representation_kuzushiji, kuzushiji_clusters,
                          markdown):
            plt.rcParams["figure.figsize"] = (14, 8)
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

            ax1.scatter(representation_emnist[:, 0], representation_emnist[:, 1], c=emnist_clusters, s=5, alpha=0.75,
                        cmap='nipy_spectral')
            ax1.axis('off')
            ax1.title.set_text('Emnist')

            ax2.scatter(representation_kuzushiji[:, 0], representation_kuzushiji[:, 1], c=kuzushiji_clusters, s=5,
                        alpha=0.75,
                        cmap='nipy_spectral')
            ax2.axis('off')
            ax2.title.set_text('Kuzushiji')

            st.markdown(markdown, unsafe_allow_html=True)
            st.pyplot(fig)


        if 'Spectral clustering' in clustering_algorithms:
            with st.spinner('Searching for optimal number of clusters via Spectral clustering'):
                model_emnist = SpectralClustering(affinity="nearest_neighbors", eigen_solver='amg', verbose=False,
                                                  n_jobs=-1)  # assign_labels='cluster_qr',
                visualizer_emnist = KElbowVisualizer(model_emnist, k=(min_cluster, max_cluster))

                model_kuzushiji = SpectralClustering(affinity="nearest_neighbors", eigen_solver='amg', verbose=False,
                                                     n_jobs=-1)
                visualizer_kuzushiji = KElbowVisualizer(model_kuzushiji, k=(min_cluster, max_cluster))

                fig, ax = plt.subplots()
                visualizer_emnist.fit(emnist_scaled)
                emnist_elbow_value_spectral = visualizer_emnist.elbow_value_
                st.markdown("<h5 style='text-align: center; color: white;'>Emnist</h5>",
                            unsafe_allow_html=True)
                ax = visualizer_emnist.show()
                st.pyplot(fig)  # Finalize and render the figure

                fig, ax = plt.subplots()
                visualizer_kuzushiji.fit(kuzushiji_scaled)
                kuzushiji_elbow_value_spectral = visualizer_kuzushiji.elbow_value_
                st.markdown("<h5 style='text-align: center; color: white;'>Kuzushiji</h5>",
                            unsafe_allow_html=True)
                ax = visualizer_kuzushiji.show()
                st.pyplot(fig)

                model_emnist = SpectralClustering(n_clusters=emnist_elbow_value_spectral, affinity="nearest_neighbors",
                                                  eigen_solver='amg', verbose=False,
                                                  n_jobs=-1)  # assign_labels='cluster_qr',

                model_kuzushiji = SpectralClustering(n_clusters=kuzushiji_elbow_value_spectral,
                                                     affinity="nearest_neighbors", eigen_solver='amg', verbose=False,
                                                     n_jobs=-1)  # assign_labels='cluster_qr',

                emnist_clusters_spectral = model_emnist.fit_predict(emnist_scaled)
                kuzushiji_clusters_spectral = model_kuzushiji.fit_predict(kuzushiji_scaled)
            st.success('Found optimal number of clusters from Spectral clustering!')

        if 'UMAP' in dimensionality_reductions and 'Spectral clustering' in clustering_algorithms:
            plot_Spectral(emnist_umap, emnist_clusters_spectral, kuzushiji_umap, kuzushiji_clusters_spectral,
                          "<h5 style='text-align: center; color: white;'>Spectral clustering with UMAP representations</h5>")

        if 'TSNE' in dimensionality_reductions and 'Spectral clustering' in clustering_algorithms:
            plot_Spectral(emnist_tsne, emnist_clusters_spectral, kuzushiji_tsne, kuzushiji_clusters_spectral,
                          "<h5 style='text-align: center; color: white;'>Spectral clustering with TSNE representations</h5>")

        if 'MDS' in dimensionality_reductions and 'Spectral clustering' in clustering_algorithms:
            plot_Spectral(emnist_mds, emnist_clusters_spectral, kuzushiji_mds, kuzushiji_clusters_spectral,
                          "<h5 style='text-align: center; color: white;'>Spectral clustering with MDS representations</h5>")

        if 'KMeans' in clustering_algorithms:
            with st.spinner('Searching for optimal number of clusters via KMeans'):
                model_emnist = KMeans()
                visualizer_emnist = KElbowVisualizer(model_emnist, k=(min_cluster, max_cluster))

                model_kuzushiji = KMeans()
                visualizer_kuzushiji = KElbowVisualizer(model_kuzushiji, k=(min_cluster, max_cluster))

                fig, ax = plt.subplots()
                visualizer_emnist.fit(emnist_scaled)
                emnist_elbow_value_kmeans = visualizer_emnist.elbow_value_
                st.markdown("<h5 style='text-align: center; color: white;'>Emnist</h5>",
                            unsafe_allow_html=True)
                ax = visualizer_emnist.show()
                st.pyplot(fig)  # Finalize and render the figure

                fig, ax = plt.subplots()
                visualizer_kuzushiji.fit(kuzushiji_scaled)
                kuzushiji_elbow_value_kmeans = visualizer_kuzushiji.elbow_value_
                st.markdown("<h5 style='text-align: center; color: white;'>Kuzushiji</h5>",
                            unsafe_allow_html=True)
                ax = visualizer_kuzushiji.show()
                st.pyplot(fig)

                model_emnist = KMeans(n_clusters=emnist_elbow_value_kmeans)
                model_kuzushiji = KMeans(n_clusters=kuzushiji_elbow_value_kmeans)

                emnist_clusters_kmeans = model_emnist.fit_predict(emnist_scaled)
                kuzushiji_clusters_kmeans = model_kuzushiji.fit_predict(kuzushiji_scaled)
            st.success('Found optimal number of clusters from KMeans clustering!')

        if 'UMAP' in dimensionality_reductions and 'KMeans' in clustering_algorithms:
            plot_Spectral(emnist_umap, emnist_clusters_kmeans, kuzushiji_umap, kuzushiji_clusters_kmeans,
                          "<h5 style='text-align: center; color: white;'>KMeans with UMAP representations</h5>")

        if 'TSNE' in dimensionality_reductions and 'KMeans' in clustering_algorithms:
            plot_Spectral(emnist_tsne, emnist_clusters_kmeans, kuzushiji_tsne, kuzushiji_clusters_kmeans,
                          "<h5 style='text-align: center; color: white;'>KMeans with TSNE representations</h5>")

        if 'MDS' in dimensionality_reductions and 'KMeans' in clustering_algorithms:
            plot_Spectral(emnist_mds, emnist_clusters_kmeans, kuzushiji_mds, kuzushiji_clusters_kmeans,
                          "<h5 style='text-align: center; color: white;'>KMeans with MDS representations</h5>")


        def plot_DBSCAN(representation_emnist, representation_kuzushiji, markdown):
            plt.rcParams["figure.figsize"] = (14, 8)
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

            clustering_emnist = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emnist_scaled)
            ax1.scatter(representation_emnist[:, 0], representation_emnist[:, 1], c=clustering_emnist, s=5, alpha=0.75,
                        cmap='nipy_spectral')
            ax1.axis('off')
            ax1.title.set_text('Emnist')

            clustering_kuzushiji = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(kuzushiji_scaled)
            ax2.scatter(representation_kuzushiji[:, 0], representation_kuzushiji[:, 1], c=clustering_kuzushiji, s=5,
                        alpha=0.75,
                        cmap='nipy_spectral')
            ax2.axis('off')
            ax2.title.set_text('Kuzushiji')

            st.markdown(markdown, unsafe_allow_html=True)
            st.pyplot(fig)


        if 'UMAP' in dimensionality_reductions and 'DBSCAN' in clustering_algorithms:
            plot_DBSCAN(emnist_umap, kuzushiji_umap,
                        "<h5 style='text-align: center; color: white;'>DBSCAN clustering with UMAP representations</h5>")

        if 'TSNE' in dimensionality_reductions and 'DBSCAN' in clustering_algorithms:
            plot_DBSCAN(emnist_tsne, kuzushiji_tsne,
                        "<h5 style='text-align: center; color: white;'>DBSCAN clustering with TSNE representations</h5>")

        if 'MDS' in dimensionality_reductions and 'DBSCAN' in clustering_algorithms:
            plot_DBSCAN(emnist_mds, kuzushiji_mds,
                        "<h5 style='text-align: center; color: white;'>DBSCAN clustering with MDS representations</h5>")


        def plot_GM(representation_emnist, emnist_elbow_value, representation_kuzushiji, kuzushiji_elbow_value,
                    markdown):
            plt.rcParams["figure.figsize"] = (14, 8)
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

            clustering_emnist = GaussianMixture(n_components=emnist_elbow_value).fit_predict(emnist_scaled)
            ax1.scatter(representation_emnist[:, 0], representation_emnist[:, 1], c=clustering_emnist, s=5, alpha=0.75,
                        cmap='nipy_spectral')
            ax1.axis('off')
            ax1.title.set_text('Emnist')

            clustering_kuzushiji = GaussianMixture(n_components=kuzushiji_elbow_value).fit_predict(kuzushiji_scaled)
            ax2.scatter(representation_kuzushiji[:, 0], representation_kuzushiji[:, 1], c=clustering_kuzushiji, s=5,
                        alpha=0.75,
                        cmap='nipy_spectral')
            ax2.axis('off')
            ax2.title.set_text('Kuzushiji')

            st.markdown(markdown, unsafe_allow_html=True)
            st.pyplot(fig)


        if 'Gaussian Mixture' in clustering_algorithms:
            with st.spinner('Gaussian Mixture search in progress'):
                n_components = range(min_cluster, max_cluster)
                score_emnist = []
                for n_comp in n_components:
                    gmm = GaussianMixture(n_components=n_comp, covariance_type='spherical')
                    gmm.fit(emnist_scaled)
                    score_emnist.append(gmm.bic(emnist_scaled))

                score_kuzushiji = []
                for n_comp in n_components:
                    gmm = GaussianMixture(n_components=n_comp, covariance_type='spherical')
                    gmm.fit(kuzushiji_scaled)
                    score_kuzushiji.append(gmm.bic(kuzushiji_scaled))

                fig, ax = plt.subplots()
                ax.plot(n_components, score_emnist, c='blue', label='emnist score')
                ax.plot(n_components, score_kuzushiji, c='red', label='kuzushiji score')
                plt.yticks([])
                plt.ylabel('Score')
                ax.legend()
                plt.xticks(n_components)
                st.markdown(
                    "<h5 style='text-align: center; color: white;'>Gaussian Mixture search for optimal number of clusters</h5>",
                    unsafe_allow_html=True)
                st.pyplot(fig)

                suggested_n_clusters = []
                for idx in range(len(score_kuzushiji)):
                    try:
                        n_clusters = n_components[idx]
                        emnist_difference = round(score_emnist[idx + 1] - score_emnist[idx])
                        kuzushiji_difference = round(score_kuzushiji[idx + 1] - score_kuzushiji[idx])
                        if emnist_difference > 0 and kuzushiji_difference > 0:
                            suggested_n_clusters.append(n_clusters)
                    except:
                        pass
                median_n_clusters = suggested_n_clusters[int(len(suggested_n_clusters) / 2)]
                f'Suggested number of clusters from GM search: {suggested_n_clusters}'
                f'Median number of clusters from GM search: {median_n_clusters}'

            st.success('Found optimal number of clusters from Gaussian Mixture search!')

            with st.spinner('Clustering data via Gaussian Mixture'):
                if 'UMAP' in dimensionality_reductions and 'Gaussian Mixture' in clustering_algorithms:
                    plot_GM(emnist_umap, median_n_clusters, kuzushiji_umap, median_n_clusters,
                            "<h5 style='text-align: center; color: white;'>GM clustering with elbow value=" + str(
                                median_n_clusters) + " from GM search on UMAP representations</h5>")

                if 'TSNE' in dimensionality_reductions and 'Gaussian Mixture' in clustering_algorithms:
                    plot_GM(emnist_tsne, median_n_clusters, kuzushiji_tsne, median_n_clusters,
                            "<h5 style='text-align: center; color: white;'>GM clustering with elbow value=" + str(
                                median_n_clusters) + " from GM search on TSNE representations</h5>")

                if 'MDS' in dimensionality_reductions and 'Gaussian Mixture' in clustering_algorithms:
                    plot_GM(emnist_mds, median_n_clusters, kuzushiji_mds, median_n_clusters,
                            "<h5 style='text-align: center; color: white;'>GM clustering with elbow value=" + str(
                                median_n_clusters) + " from GM search on MDS representations</h5>")
            st.success('Successfully plotted all Gaussian Mixture predictions!')


        def plot_histogram(emnist_clusters, kuzushiji_clusters, markdown):
            unique_kuzushiji, counts_kuzushiji = np.unique(emnist_clusters, return_counts=True)
            unique_emnist, counts_emnist = np.unique(kuzushiji_clusters, return_counts=True)

            unique_emnist = unique_emnist.astype('str')
            unique_kuzushiji = unique_kuzushiji.astype('str')

            unique_kuzushiji, counts_kuzushiji = zip(
                *sorted(zip(unique_kuzushiji, counts_kuzushiji), key=lambda x: x[1], reverse=True))
            unique_emnist, counts_emnist = zip(
                *sorted(zip(unique_emnist, counts_emnist), key=lambda x: x[1], reverse=True))

            plt.rcParams["figure.figsize"] = (8, 8)

            fig, ax = plt.subplots()
            ax.axis('off')
            ax.bar(range(len(counts_emnist)), counts_emnist, color='blue', label='emnist', alpha=0.6)
            ax.bar(range(len(counts_kuzushiji)), counts_kuzushiji, color='red', label='kuzushiji', alpha=0.6)

            st.markdown(markdown, unsafe_allow_html=True)
            st.pyplot(fig)


        with st.spinner('Generating histograms'):
            if 'Spectral clustering' in clustering_algorithms:
                plot_histogram(emnist_clusters_spectral, kuzushiji_clusters_spectral,
                               "<h5 style='text-align: center; color: white;'>Spectral clustering cluster histogram</h5>")

            if 'KMeans' in clustering_algorithms:
                plot_histogram(emnist_clusters_kmeans, kuzushiji_clusters_kmeans,
                               "<h5 style='text-align: center; color: white;'>KMeans cluster histogram</h5>")

            if 'DBSCAN' in clustering_algorithms:
                clustering_emnist = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emnist_scaled)
                clustering_kuzushiji = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(kuzushiji_scaled)
                plot_histogram(clustering_emnist, clustering_kuzushiji,
                               "<h5 style='text-align: center; color: white;'>DBSCAN cluster histogram</h5>")

            if 'Gaussian Mixture' in clustering_algorithms:
                clustering_emnist = GaussianMixture(n_components=median_n_clusters).fit_predict(emnist_scaled)
                clustering_kuzushiji = GaussianMixture(n_components=median_n_clusters).fit_predict(kuzushiji_scaled)
                plot_histogram(clustering_emnist, clustering_kuzushiji,
                               "<h5 style='text-align: center; color: white;'>Gaussian Mixture cluster histogram</h5>")

        st.success('Histograms generated successfully')
        st.balloons()

elif selection == 'Generating translated pages':

    n_clusters = st.number_input('Insert number of clusters', min_value=0, max_value=100, step=1, value=50)
    algorithm = st.selectbox(
        'Which clustering algorithm do you wish to use?',
        ('Gaussian Mixture', 'Spectral clustering', 'KMeans', 'DBSCAN'))
    type_of_letters = st.selectbox(
        'Which type of letters do you wish to use?',
        ('Passed through autoencoder', 'Input data'))

    if 'DBSCAN' == algorithm:
        eps = st.slider('Choose number of eps for DBSCAN', 0.0, 2.0, 0.1)
        min_samples = st.slider('Choose number of minimum samples for DBSCAN', 0, 100, 5)

    col1_1, col2_1, col3_1, col4_1, col5_1 = st.columns(5)
    if col3_1.button('Engage translation'):
        path = "data/encoded_data/1-2-3-4-5/trial_4"

        with st.spinner('Loading data in progress'):
            emnist_pred = np.load(ENCODED_EMNIST_PATH)['arr_0']
            emnist_scaled = StandardScaler().fit_transform(emnist_pred)
            kuzushiji_pred = np.load(ENCODED_KUZUSHIJI_PATH)['arr_0']
            kuzushiji_scaled = StandardScaler().fit_transform(kuzushiji_pred)
            decoded_emnist_pred = np.load(DECODED_EMNIST)['arr_0'] #TODO pick correct path here
        st.success('Loading data done!')

        with st.spinner('Asserting correct data shape'):
            assert load_emnist_pages(5, trial='trial_4').shape[0] == emnist_pred.shape[0]
            assert load_kuzushiji_pages(5, trial='trial_4').shape[0] == kuzushiji_pred.shape[0]
        st.success('Data shape from loaded pages matches representation shape!')

        with st.spinner('Clustering in progress'):
            if 'Gaussian Mixture' == algorithm:
                clustering_emnist = GaussianMixture(n_components=n_clusters).fit_predict(emnist_scaled)
                clustering_kuzushiji = GaussianMixture(n_components=n_clusters).fit_predict(kuzushiji_scaled)

            if 'KMeans' == algorithm:
                clustering_emnist = KMeans(n_clusters=n_clusters).fit_predict(emnist_scaled)
                clustering_kuzushiji = KMeans(n_clusters=n_clusters).fit_predict(kuzushiji_scaled)

            if 'Spectral clustering' == algorithm:
                clustering_emnist = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='cluster_qr', eigen_solver='amg', verbose=True, n_jobs=-1).fit_predict(emnist_scaled)
                clustering_kuzushiji = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='cluster_qr', eigen_solver='amg', verbose=True, n_jobs=-1).fit_predict(kuzushiji_scaled)

            if 'DBSCAN' == algorithm:
                clustering_emnist = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emnist_scaled)
                clustering_kuzushiji = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(kuzushiji_scaled)
        st.success('Clustering done')

        with st.spinner('Matching clusters in progress'):
            unique_kuzushiji, counts_kuzushiji = np.unique(clustering_kuzushiji, return_counts=True)
            unique_emnist, counts_emnist = np.unique(clustering_emnist, return_counts=True)

            unique_emnist = unique_emnist.astype('str')
            unique_kuzushiji = unique_kuzushiji.astype('str')

            unique_kuzushiji, counts_kuzushiji = zip(
                *sorted(zip(unique_kuzushiji, counts_kuzushiji), key=lambda x: x[1], reverse=True))
            unique_emnist, counts_emnist = zip(
                *sorted(zip(unique_emnist, counts_emnist), key=lambda x: x[1], reverse=True))

            fig, ax = plt.subplots()
            plt.rcParams["figure.figsize"] = (8, 8)

            plt.axis('off')
            ax.bar(range(len(counts_emnist)), counts_emnist, color='blue', label='emnist', alpha=0.6)
            ax.bar(range(len(counts_kuzushiji)), counts_kuzushiji, color='red', label='kuzushiji', alpha=0.6)
            st.pyplot(fig)
        st.success('Clusters matched successfully')

        with st.spinner('Generating translation'):
            kuzushiji_pages, kuzushiji_spaces = load_kuzushiji_pages_with_spaces(5, trial='trial_4')
            emnist_pages, emnist_spaces = load_emnist_pages_with_spaces(5, trial='trial_4')
            emnist_idx_clusters = dict()
            kuzushiji_idx_clusters = dict()

            for idx in range(len(emnist_pages)):
                try:
                    emnist_idx_clusters[clustering_emnist[idx]].append(idx)
                except:
                    emnist_idx_clusters[clustering_emnist[idx]] = list()
                    emnist_idx_clusters[clustering_emnist[idx]].append(idx)

            for idx in range(len(kuzushiji_pages)):
                try:
                    kuzushiji_idx_clusters[clustering_kuzushiji[idx]].append(idx)
                except:
                    kuzushiji_idx_clusters[clustering_kuzushiji[idx]] = list()
                    kuzushiji_idx_clusters[clustering_kuzushiji[idx]].append(idx)

            kuzushiji_to_emnist = dict()
            emnist_to_kuzushiji = dict()

            for idx in range(len(unique_kuzushiji)):
                kuzushiji_to_emnist[unique_kuzushiji[idx]] = unique_emnist[idx]
                emnist_to_kuzushiji[unique_emnist[idx]] = unique_kuzushiji[idx]

            kuzushiji_translated_emnist = []
            for element_of_cluster in clustering_kuzushiji:
                kuzushiji_translated_emnist.append(kuzushiji_to_emnist[str(element_of_cluster)])

            picture_vector = []
            for cluster in kuzushiji_translated_emnist:
                if type_of_letters == 'Input data':
                    picture_vector.append(emnist_pages[np.random.choice(emnist_idx_clusters[int(cluster)])])
                if type_of_letters == 'Passed through autoencoder':
                    picture_vector.append(decoded_emnist_pred[np.random.choice(emnist_idx_clusters[int(cluster)])])

            columns = 80
            rows = 114
            cell = 32
            max_chars = columns * rows

            empty_sheet = np.zeros(shape=(rows * cell, columns * cell))

            l_idx = 0
            translated_emnist_sheets = list()
            iter_picture_vector = iter(picture_vector)

            while True:
                if l_idx == (len(picture_vector) + len(kuzushiji_spaces)):
                    break

                sheet = copy.deepcopy(empty_sheet)

                for i in range(0, sheet.shape[0], 32):
                    for j in range(0, sheet.shape[1], 32):
                        if l_idx in kuzushiji_spaces:
                            sheet[i:i + 32, j:j + 32] = (np.zeros(shape=(32, 32)))
                        else:
                            sheet[i:i + 32, j:j + 32] = next(iter_picture_vector)

                        l_idx += 1

                translated_emnist_sheets.append(sheet)

            path = 'data/translated_pages/streamlit/emnist/'

            for idx in range(len(translated_emnist_sheets)):
                img = cv2.convertScaleAbs(translated_emnist_sheets[idx], alpha=255.0)
                cv2.imwrite(path + "/emnist_" + str(idx) + ".png", img)

        st.success('Pages successfully translated!')

        col1_2, col2_2, col3_2 = st.columns(3)
        col2_2.text('Comparing random pages')

        plt.rcParams["figure.figsize"] = (8, 8)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        rng = np.random.randint(0, 5)
        original = mpimg.imread('data/generated/streamlit/emnist/emnist_' + str(rng) + '.png')
        ax1.imshow(original)
        ax1.axis('off')

        translated = mpimg.imread('data/translated_pages/streamlit/emnist/emnist_' + str(rng) + '.png')
        ax2.imshow(translated)
        ax2.axis('off')

        st.pyplot(fig)
        st.balloons()
