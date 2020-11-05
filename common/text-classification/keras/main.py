"""
0;95;0cNote: This sample code is adapted from this great tutorial:
https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/

Data used in this tutorial can be downloaded from:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/review_polarity.tar.gz
"""


import argparse
import os
from collections import Counter
from string import punctuation
from typing import List

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords


DATASET_PATH = '/Users/timowang/Downloads/review_polarity/txt_sentoken/'


def load_text(file_path: str) -> str:
    with open(file_path) as fp:
        text = fp.read()
    return text
    

def clean_text(text: str) -> List[str]:
    tokens = word_tokenize(text)
    punkt_idx2null = str.maketrans('', '', punctuation)
    tokens = [token.translate(punkt_idx2null) for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # tokens = [token for token in tokens if len(token) > 1]
    return ' '.join(tokens)


def process_texts(file_dir: str, is_train: bool) -> None:
    all_texts = []
    for file_name in os.listdir(file_dir):
        if is_train and file_name.startswith('cv9'):
            continue
        if not is_train and not file_name.startswith('cv9'):
            continue
        file_path = os.path.join(file_dir, file_name)
        cleaned_text = clean_text(load_text(file_path))
        all_texts.append(cleaned_text)
    return all_texts


def build_cnn_classifier(vocab_size: int, max_length: int, emb_size=128, num_filters=32, kernel_size=8) -> Sequential:
    model = Sequential()
    model.add(Embedding(vocab_size, emb_size, input_length=max_length)) # max_len x emb_size
    
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    file_path = os.path.join(DATASET_PATH, 'pos/cv000_29590.txt')
    text = load_text(file_path)
    text = clean_text(text)
    print(text)

    train_texts = process_texts(os.path.join(DATASET_PATH, 'neg'), True) + \
                  process_texts(os.path.join(DATASET_PATH, 'pos'), True)
    test_texts = process_texts(os.path.join(DATASET_PATH, 'neg'), False) + \
                 process_texts(os.path.join(DATASET_PATH, 'pos'), False)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    max_length = max([len(s.split()) for s in train_texts])

    encoded_train_texts = tokenizer.texts_to_sequences(train_texts)    
    X_train = pad_sequences(encoded_train_texts, maxlen=max_length, padding='post')
    y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])
    
    encoded_test_texts = tokenizer.texts_to_sequences(test_texts)
    X_test = pad_sequences(encoded_test_texts, maxlen=max_length, padding='post')
    y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])
    
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)
    
    model = build_cnn_classifier(vocab_size, max_length)
    model.fit(X_train, y_train, epochs=10, verbose=2)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy: %f' % acc)
    
    

if __name__ == '__main__':
    main()
