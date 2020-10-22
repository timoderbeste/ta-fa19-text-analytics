"""
Note: This sample code is adapted from this great tutorial:
https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/

Data used in this tutorial can be downloaded from:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/review_polarity.tar.gz
"""


import argparse
import os
from collections import Counter
from string import punctuation
from typing import List

import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Dropout, Embedding, Flatten, GlobalAveragePooling1D, Input, Layer, LayerNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords


DATASET_PATH = '/Users/timowang/Downloads/review_polarity/txt_sentoken/'


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
    
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


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
    all_tokens = []
    for file_name in os.listdir(file_dir):
        if is_train and file_name.startswith('cv9'):
            continue
        if not is_train and not file_name.startswith('cv9'):
            continue
        file_path = os.path.join(file_dir, file_name)
        tokens = clean_text(load_text(file_path))
        all_tokens.append(tokens)
    return all_tokens


def build_cnn_classifier(vocab_size: int, max_length: int, emb_size=128, num_filters=32, kernel_size=8) -> Sequential:
    model = Sequential()
    model.add(Embedding(vocab_size, emb_size, input_length=max_length))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_transformer_classifier(vocab_size: int, max_length: int, emb_size=128):
    inputs = Input(shape=(max_length,))
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, emb_size)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(emb_size, 8, 128)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    file_path = os.path.join(DATASET_PATH, 'pos/cv000_29590.txt')
    text = load_text(file_path)
    tokens = clean_text(text)
    print(tokens)
    
    train_texts = process_texts(os.path.join(DATASET_PATH, 'pos'), True) + \
                   process_texts(os.path.join(DATASET_PATH, 'neg'), True)
    test_texts = process_texts(os.path.join(DATASET_PATH, 'pos'), False) + \
                 process_texts(os.path.join(DATASET_PATH, 'neg'), False)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    
    encoded_train_texts = tokenizer.texts_to_sequences(train_texts)
    max_length = max([len(s.split()) for s in train_texts])
    X_train = pad_sequences(encoded_train_texts, maxlen=max_length, padding='post')
    y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])
    
    encoded_test_texts = tokenizer.texts_to_sequences(test_texts)
    X_test = pad_sequences(encoded_test_texts, maxlen=max_length, padding='post')
    y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])
    
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)
    
    # model = build_classifier(vocab_size, max_length)
    model = build_transformer_classifier(vocab_size, max_length)
    model.fit(X_train, y_train, epochs=10, verbose=2)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy: %f' % acc)
    
    

if __name__ == '__main__':
    main()
