from typing import List


class Vocab:
    def __init__(self):
        self.token2idx = {
            'UNK': 0,
        }
        self.idx2token = {
            0: 'UNK',
        }
        self.count = 1
        
    def add_token(self, token: str):
        if token not in self.token2idx:
            self.token2idx[token] = self.count
            self.idx2token[self.count] = token
            self.count += 1
        
    def add_tokens(self, tokens: List[str]):
        for token in tokens:
            self.add_token(token)