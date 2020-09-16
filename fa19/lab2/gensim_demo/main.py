import argparse

import nltk
from gensim.models import Word2Vec


def split_sentences(lines: [str]) -> [[str]]:
    sents = []
    for line in lines:
        tokens = []
        splitted_lines = nltk.sent_tokenize(line)
        for splitted_line in splitted_lines:
            tokens.append(nltk.word_tokenize(splitted_line))
        sents.extend(tokens)
    return sents


def load_lines(file_name: str) -> [str]:
    lines = []
    with open(file_name, 'r', encoding='utf-8', errors='replace') as fp:
        for line in fp:
            lines.append(line)
    return lines


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_file', type=str, required=True)
    arg_parser.add_argument('-do_eval', action='store_true')
    args = arg_parser.parse_args()
    input_file = args.input_file
    do_eval = args.do_eval

    if not do_eval:
        sents = (split_sentences(load_lines(input_file)))
        model = Word2Vec(sents, min_count=10, size=50, workers=10, window=5, sg=1)
        model.save(fname_or_handle='/Users/timowang/Entwickler/results/word2vec-gensim/model')
    else:
        model = Word2Vec.load(fname_or_handle='/Users/timowang/Entwickler/results/word2vec-gensim/model')
        while True:
            user_input = input('Please enter a word to find its similarity. Enter EXIT to quit.\n')
            if user_input == 'EXIT':
                break
            if user_input not in model:
                print('%s cannot be found. Please try another word!' % user_input)
                continue
            print('The word vector of %s:' % user_input)
            print(model[user_input])


if __name__ == '__main__':
    main()