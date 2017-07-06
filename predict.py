import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec

import re
from bs4 import BeautifulSoup

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model


def load_data(type, labels=True):

    data_folder = os.path.join(os.getcwd(), 'data')

    if type == 'train':
        if labels:
            filename = 'labeledTrainData.tsv'
        elif not labels:
            filename = 'unlabeledTrainData.tsv'
    elif type == 'test':
        filename = 'testData.tsv'

    filepath = os.path.join(data_folder, filename)

    data = pd.read_csv(filepath, sep='\t')

    return data


def preprocessor(rawdata, labels=True):

    reviews = rawdata['review']
    reviews = np.array(reviews)

    id = rawdata['id']
    id = np.array(id, dtype=np.str)

    if labels:
        target = rawdata['sentiment']
        target = np.array(target, dtype=np.uint8)

        return reviews, target, id
    else:

        return reviews, id

def textcleaner(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

train_labeled = load_data(type='train', labels=True)
train_full = pd.DataFrame(train_labeled)

x_train, y_train, id_train = preprocessor(rawdata=train_labeled)

model = Word2Vec(x_train)