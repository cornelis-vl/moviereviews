# Imports
import numpy as np
import pandas as pd
import os
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model

import nltk.data
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords

from gensim.models import word2vec


# Functions
def load_data(type, labels=True):
    data_folder = os.path.join(os.getcwd(), 'data', 'text')

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


def clean(text, remove_stopwords=False):
    review_text = bs(text).get_text()

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return words


def textcleaner(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def clean_all(clean_func, data):
    cleaned_data = [clean_func(x) for x in data]
    return cleaned_data


def train_test_split(features, target, identifier, prop=0.85):
    """Split full dataset in train and test data according to specified property
    assigned to the train data.
    """

    # Shuffle rows to mimic random selection
    rows = np.arange(0, target.shape[0])
    np.random.shuffle(rows)

    # Define threshold to split shuffled rows into train and test sets
    threshold = int(prop * len(rows))
    train_rows = rows[0:threshold]
    test_rows = rows[threshold:len(rows)]

    # Create train and test sets for features, target and ids
    train_id = identifier[train_rows]
    test_id = identifier[test_rows]
    train_features = features[train_rows]
    test_features = features[test_rows]
    train_target = target[train_rows]
    test_target = target[test_rows]

    return train_features, test_features, train_target, test_target, train_id, test_id


def review_to_sentences(review, remove_stopwords=False):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( clean( raw_sentence, remove_stopwords ))

    return sentences


def all_r2s(data):
    sentences = []
    for x in data:
        sentences += review_to_sentences(x)

    return sentences


def create_embedding_layer(embeddings_index, word_index, embedding_dim=100, max_seq_length=1000):

    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = 0

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_seq_length,
                                trainable=True)

    return embedding_layer




def tokenize(data, max_words, max_seq_len):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)

    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    tokeniz_data = pad_sequences(sequences, maxlen=max_seq_len)

    return tokeniz_data, word_index, sequences


def build_model(embedded_sequences, sequence_input):

    l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model


def create_wordvec(sentences, embed_dim=100, min_word_count=10):

    model = word2vec.Word2Vec(sentences, workers=4, size=embed_dim, min_count=min_word_count,
                      window=10, sample=1e-3)

    word_vector = model.wv

    return word_vector

# Script

if __name__ == 'main':

    print('main')

else:


    # Load
    train_labeled = load_data(type='train', labels=True)
    train_full = pd.DataFrame(train_labeled)

    # Structure and clean
    x_train, y_train, id_train = preprocessor(rawdata=train_labeled)
    x_train_clean = clean_all(textcleaner, x_train)

    # Tokenize
    x_train_tok, word_ind, sequences = tokenize(data=x_train_clean, max_words=20000, max_seq_len=1000)

    # Create word vector
    sentences = all_r2s(x_train_clean)
    wordvec = create_wordvec(sentences=sentences)

    emb_lyr = create_embedding_layer(wordvec, word_ind)

    sequence_input = Input(shape=(1000,), dtype='int32')
    embedded_sequences = emb_lyr(sequence_input)

    clf = build_model(embedded_sequences=embedded_sequences, sequence_input=sequence_input)


    labels = to_categorical(np.asarray(y_train))

    X_train_train, X_train_test, Y_train_train, \
    Y_train_test, ID_train_train, ID_train_test = train_test_split(features=x_train_tok,
                                                                   target=labels,
                                                                   identifier=id_train)

    clf.fit(X_train_train, Y_train_train, validation_data=(X_train_test, Y_train_test),
        epochs=1, batch_size=128)