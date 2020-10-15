import argparse

import nltk
from gensim.models import Word2Vec


def load_lines(file_name: str) -> [str]:
    lines = []
    with open(file_name, 'r', encoding='utf-8', errors='replace') as fp:
        for line in fp:
            lines.append(line)
    return lines


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_file', type=str, required=False)
    arg_parser.add_argument('--do_eval', action='store_true')
    args = arg_parser.parse_args()
    input_file = args.input_file
    do_eval = args.do_eval

    if not do_eval:
        if not input_file:
            print('no input file provided for training')
            exit(1)
        print('Processing input text...')
        sents = split_sentences(load_lines(input_file))
        print('Finished processing the input text. Starting to train a skip gram model...')
        print('There are %d sentences' % len(sents))
        model = Word2Vec(corpus_file='/Users/timowang/Desktop/text-analytics-demo/word2vec-master/text8', min_count=10, size=200, negative=25, iter=15, workers=10, window=8, sg=0)
        print('Model trained. Saving...')
        model.save(fname_or_handle='/Users/timowang/Desktop/text-analytics-demo/gensim_demo/model')
    else:
        model = Word2Vec.load(fname_or_handle='/Users/timowang/Desktop/text-analytics-demo/gensim_demo/model')
        while True:
            user_input = input('Please enter a word to find its embedding. Enter EXIT to quit.\n')
            if user_input == 'EXIT':
                break
            if user_input not in model:
                print('%s cannot be found. Please try another word!' % user_input)
                continue
            print('The word vector of %s:' % user_input)
            print(list(model[user_input]))


if __name__ == '__main__':
    main()
