from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

class SnowballTokenizer:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")

    def __call__(self, words):
        return [self.stemmer.stem(word) for word in word_tokenize(words)]

snowball_stem = SnowballTokenizer()
print(snowball_stem('Martin Porter has endorsed several modifications to the Porter algorithm since writing his original paper, '))
