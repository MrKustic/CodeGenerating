import argparse
import os
import pickle
import re
import numpy as np
import random


def some_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help="Input directory")
    parser.add_argument('--model', help="Output directory for model")
    return parser.parse_args()


def get_args():
    args = some_args()
    if args.input_dir:
        input_path = args.input_dir
    else:
        input_path = "stdin"
    model_path = args.model
    return input_path, model_path


class Model:

    def __init__(self, path, seed=31):
        self.cnt = {}
        self.d = {}
        self.path = path
        self.seed = seed

    def count_words(self, words):
        n = len(words)
        d = self.cnt
        for i in range(n - 2):
            key = (words[i], words[i + 1])
            nword = words[i + 2]
            if key not in d:
                d[key] = {}
            d[key][nword] = d[key].get(nword, 0) + 1
        self.cnt = d

    def set_probabilities(self):
        res = {}
        d = self.cnt
        for key in d:
            cnt = 0
            res[key] = []
            for word in d[key]:
                cnt += d[key][word]
            for word in d[key]:
                res[key].append((word, d[key][word] / cnt))
        self.d = res

    def getwords(self, f):
        s = ""
        for x in f:
            s += x
        s = s.lower()
        s = re.sub(r'[^а-я-ё\n ]', '', s)
        words = [i for i in s.split() if i[0] != '-' and i[-1] != '-' and i.count('-') <= 1]
        return words

    def fit(self):
        for address, dirs, files in os.walk(self.path):
            for name in files:
                try:
                    with open(os.path.join(address, name), 'r', encoding='utf-8') as f:
                        words = self.getwords(f)
                except UnicodeDecodeError:
                    with open(os.path.join(address, name), 'r') as f:
                        words = self.getwords(f)
                self.count_words(words)
        self.set_probabilities()

    def generate_random_word(self):
        nword = np.random.choice(random.choice(list(self.d)))
        return nword

    def generate(self, prefix, length):
        result = prefix.split()
        while len(result) < 2:
            nword = self.generate_random_word()
            result.append(nword)
        while len(result) < length:
            key = (result[-2], result[-1])
            if key not in self.d:
                nword = self.generate_random_word()
            else:
                words, probs = zip(*self.d[key])
                nword = np.random.choice(words, 1, True, probs)[0]
            result.append(nword)
        return result


def main():
    input_path, model_path = get_args()
    model = Model(input_path)
    model.fit()
    pickle.dump(model, open(model_path, "wb"))


if __name__ == '__main__':
    main()
