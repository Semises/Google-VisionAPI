import os, io
import pandas as pd
import csv
from google.cloud import vision_v1
from google.cloud.vision_v1 import types

# Klucz weryfikacyjny
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'

# Stworzenie sesji
client = vision_v1.ImageAnnotatorClient()

# Wybor sciezki i nazwy analizowanego pliku
path = r'C:\Users\Daniel\Documents\PythonVenv\VisionAPIDemo\Images'
f = 'setagaya_small.jpeg'

# Funkcja wykrywania tekstu
def detectText(img):
    # Powiazanie sciezki i nazwy obrazu
    with io.open(img, 'rb') as image_file:
        content = image_file.read()

    # Tworzy wystapienie obrazu
    image = vision_v1.types.Image(content=content)

    # Opisuje odpowiedz obrazu
    response = client.text_detection(image=image)
    df = pd.DataFrame(columns = ['locale', 'description'])
    texts = response.text_annotations

    # Dodaje wykryty tekst do ramki danych
    for text in texts:
        df = df.append(
            dict(
                locale = text.locale,
                description = text.description
            ),
            ignore_index = True
        )
    return df

# Funkcja klasyfikacji zdjecia
def detectLabels(img):
    with io.open(img, 'rb') as image_file:
        content = image_file.read()
    # Tworzy wystapienie obrazu
    image = vision_v1.types.Image(content=content)

    # Opisuje odpowiedz obrazu
    response = client.text_detection(image=image)
    labels = response.label_annotations
    df = pd.DataFrame(columns = ['description', 'score', 'topicality'])

    # Dodaje wykryta klasyfikacje do ramki danych
    for label in labels:
        df = df.append(
            dict(
                description = label.description,
                score = label.score,
                topicality = label.topicality
            ),
            ignore_index=True
        )
    return df


# Funkcja zapisywania ramek danych do pliku CSV
def writeDataFrameToCSV(path, f):
    # Wybiera sciezke i nazwe pliku ze zmiennych globalnych
    FOLDER_PATH = path
    FILE_NAME = f

    # Zapisuje ramke danych
    dataframe_1 = detectText(os.path.join(FOLDER_PATH, FILE_NAME))
    dataframe_2 = detectLabels(os.path.join(FOLDER_PATH, FILE_NAME))

    # Zapisuje ramke danych jako plik CSV
    dataframe_1.to_csv("texts.csv")
    dataframe_2.to_csv("labels.csv")

writeDataFrameToCSV(path, f)
        