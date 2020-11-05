"""
This sample code is adapted from this great tutorial:
https://www.aitimejournal.com/@akshay.chavan/complete-tutorial-on-named-entity-recognition-ner-using-python-and-keras
It also has an accompanying explanation on CRF:
https://www.aitimejournal.com/@akshay.chavan/introduction-to-conditional-random-fields-crfs

You may also find the following explanation Keras helpful.
https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
"""

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from tf2crf import CRF

from utils import load_sentences, load_vocab

DATA_PATH = '/Users/timowang/Downloads/ner_dataset.csv'

BATCH_SIZE = 64
EPOCHS = 8
MAX_LEN = 75
EMBEDDING_SIZE = 40


def load_dataset(data_path: str):
    sentences = load_sentences(data_path)
    word2idx, tag2idx, idx2word, idx2tag, words, tags, num_tags = load_vocab(data_path)
    
    X = [[word2idx[tup[0]] for tup in sentence] for sentence in sentences]
    X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding='post', value=word2idx['PAD'])
    y = [[tag2idx[tup[2]] for tup in sentence] for sentence in sentences]
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding='post', value=tag2idx['PAD'])
    y = [to_categorical(i, num_classes=num_tags + 1) for i in y]
    return X, y, words, tags


def main():
    X, y, words, tags = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

    inp = Input(shape = (MAX_LEN,))
    model = Embedding(input_dim = len(words) + 2, output_dim = EMBEDDING_SIZE, input_length = MAX_LEN, mask_zero = True)(inp)
    model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(50, activation="relu"))(model)
    crf = CRF(len(tags) + 1)  # CRF layer
    out = crf(model)  # output
    
    model = Model(inp, out)
    model.compile(optimizer="rmsprop", loss=crf.loss, metrics=[crf.accuracy])
    
    model.summary()

    checkpointer = ModelCheckpoint(filepath = 'model.h5',
                                   verbose = 0,
                                   mode = 'auto',
                                   save_best_only = True,
                                   monitor='val_loss')

    history = model.fit(X_train, np.array(y_train), batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_split=0.1, callbacks=[checkpointer])
    
    


if __name__ == '__main__':
    main()
