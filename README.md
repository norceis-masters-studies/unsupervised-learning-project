# Uczenie nienadzorowane - projekt

### Wymagania

**Wersja Python** - 3.10

##### Biblioteki

Wymagane biblioteki znajdują się w pliku `requirements.txt`, znajdującym się w katalogu głównym projektu.

###### Pip
Aby zainstalować biblioteki za pomocą `pip`, nalezy uruchomić polecenie `pip install -r requirements.txt`


# Instrukcja

## Jupyter
### Generacja danych

Aby wygenerować dane, należy uruchomić notebook o nazwie `generate_sheets.ipynb`, znajdujący się w scieżce `scr/notebooks`. 
Po uruchomieniu, w katalogu w którym został uruchomiony notebook, powinny zostać utworzone pliki `empty_sheet_kuzushiji.png` 
dla znaków **KUZUSHIJI-49** oraz `empty_sheet_emnist.png` dla **EMNIST-BYMERGE**.

### Uczenie

Aby nauczyć autoencoder, należy uruchomić notebook o nazwie `autoencoder.ipynb`, znajdujący się w scieżce `scr/notebooks`.
Po uruchomieniu, w katalogu `data/models`, powinny zostać utworzone pliki zawierajace wagę modeli 
dla znaków **KUZUSHIJI-49** oraz **EMNIST-BYMERGE**.

### Enkodowanie

Aby zakodować dane, należy uruchomić notebook o nazwie `encoded_data_generation.ipynb`, znajdujący się w scieżce `scr/notebooks`.
Po uruchomieniu, w katalogu `data/encoded_data`, powinny zostać utworzone pliki zawierajace zakodowane dane
dla znaków **KUZUSHIJI-49** oraz **EMNIST-BYMERGE**.

### Klasteryzacja

Aby przeprowadzić klasteryzację, należy uruchomić notebook o nazwie `clustering.ipynb`, znajdujący się w scieżce `scr/notebooks`.
W pliku zaprezentowane zostały różne sposoby wizualizacji danych oraz przeszukiwanie róznymi
metodami klasteryzacji.

## Streamlit

Aby uruchomić projekt poprzez Streamlit, należy uruchomić polecenie `python -m streamlit run src/streamlit.py` 
w głównym katalogu projektu.

Uruchomiona strona prezentuje wszystkie etapy projektu, wraz z możliwością wyboru danych, modelu,
metody klasteryzacji oraz sposobu wizualizacji danych.

Poszegolne kroki sa opisane w odpowiednich sekcjach

