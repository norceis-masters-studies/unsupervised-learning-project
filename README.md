# Unsupervised learning project

The project is about mapping latin emnist letters to japanese characters called kuzushiji via autoencoder, clustering them and matching letter to letter via cluster population number to each other in order to translate it.

### Requirements

**Python** - 3.10

##### Libraries

Required libraries are in "requirements.txt".

###### Pip
To install libraries - run "pip install -r requirements.txt"

# Instructions

## Jupyter
### Data generation

To generate data, run "/src/notebooks/generate_sheets.ipynb". After running it there should be 35 pages of each character variant in /data/generated.

### Learning

To make the autoencoder learn, run "/src/notebooks/autoencoder.ipynb". After running it there should be 2 .pth files with autoencoder parameters in /data/models.

### Enkodowanie

To encode data, run "/src/notebooks/encoded_data_generation.ipynb". After running it there should be 2 .npz files, which are ancoded data representations in /data/encoded_data

### Klasteryzacja

To cluster, run "/src/notebooks/clustering.ipynb". After running it you should inspect which method do you prefer and remember your choice for generating translated pages.

## Streamlit

To run project through Streamlit, run `python -m streamlit run src/streamlit.py`. This depicts all of the project's stages.


