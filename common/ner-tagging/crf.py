"""
This sample code is adapted from this great tutorial:
https://www.aitimejournal.com/@akshay.chavan/complete-tutorial-on-named-entity-recognition-ner-using-python-and-keras
It also has an accompanying explanation on CRF:
https://www.aitimejournal.com/@akshay.chavan/introduction-to-conditional-random-fields-crfs
"""
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report

from utils import load_sentences, get_labels

DATA_PATH = '/Users/timowang/Downloads/ner_dataset.csv'


def get_token_level_features(tokens: List[Tuple[str]], idx: int) -> Dict:
    word = tokens[idx][0]
    pos = tokens[idx][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'pos': pos,
        'pos[:2]': pos[:2],
    }
    
    if idx > 0:
        left_word = tokens[idx - 1][0]
        left_pos = tokens[idx - 1][1]
        features.update({
            'left_word.lower()': left_word.lower(),
            'left_word.istitle()': left_word.istitle(),
            'left_word.isupper()': left_word.isupper(),
            'left_pos': left_pos,
        })
    else:
        features['BOS'] = True
    
    if idx < len(tokens) - 1:
        right_word = tokens[idx - 1][0]
        right_pos = tokens[idx - 1][1]
        features.update({
            'right_word.lower()': right_word.lower(),
            'right_word.istitle()': right_word.istitle(),
            'right_word.isupper()': right_word.isupper(),
            'right_pos': right_pos,
        })
    else:
        features['EOS'] = True
    
    return features


def get_sent_features(tokens: List[Tuple[str]]) -> List[Dict]:
    return [get_token_level_features(tokens, i) for i in range(len(tokens))]


def load_dataset(data_path: str):
    sentences = load_sentences(data_path)
    X = [get_sent_features(sentence) for sentence in sentences]
    y = [get_labels(sentence) for sentence in sentences]
    return X, y


def main():
    X, y = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False)
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)
    f1_score = flat_f1_score(y_test, y_pred, average='weighted')
    print(f1_score)
    
    report = flat_classification_report(y_test, y_pred)
    print(report)


if __name__ == '__main__':
    main()
