from typing import List, Tuple

import pandas as pd


def load_sentences(data_path: str) -> List[List[Tuple[str]]]:
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    df.fillna(method='ffill')
    grouped_sents = df.groupby('Sentence #').apply(
        lambda series: [(word, pos, tag) for word, pos, tag in zip(series['Word'].values.tolist(),
                                                                   series['POS'].values.tolist(),
                                                                   series['Tag'].values.tolist())]
    )
    return [s for s in grouped_sents]


def get_labels(tokens: List[Tuple[str]]) -> List[str]:
    return [label for _, _, label in tokens]


def get_words(tokens: List[Tuple[str]]) -> List[str]:
    return [word for word, _, _ in tokens]


def load_vocab(data_path: str):
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    df.fillna(method='ffill')
    words = list(df['Word'].unique())
    tags = list(df['Tag'].unique())

    word2idx = {w : i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    tag2idx = {t : i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0

    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: w for w, i in tag2idx.items()}

    num_tags = df['Tag'].nunique()
    
    return word2idx, tag2idx, idx2word, idx2tag, words, tags, num_tags