from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import numpy as np


LANGS = ['eng', 'fra', 'swe']


class Classifier:
    def __init__(self, parser):
        self.clf = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=5)
        self.vector = DictVectorizer(sparse=False)
        self.parser = parser
        
        self.vectorize()
        self.shuffle()
        self.split()
    
    def train(self):
        self.clf.fit(*self.training)

    def test(self):
        return accuracy_score(self.testing[1], self.predict(self.testing[0]))

    def predict(self, texts, readable=False):
        predictions = self.clf.predict(texts)

        if readable:
            return [LANGS[prediction] for prediction in predictions]

        return predictions

    def prepare(self, text):
        return self.vector.transform(self.parser.ngram_freq(text, 1))

    def vectorize(self):
        self.x = self.vector.fit_transform(self.parser.get('freqs'))
        self.y = [LANGS.index(lang) for lang in self.parser.get('lang')]

    def shuffle(self):
        indices = list(range(self.x.shape[0]))
        np.random.shuffle(indices)

        self.x = self.x[indices, :]
        self.y = np.array(self.y)[indices]

    def split(self):
        TRAINING_COUNT = int(self.x.shape[0] * 4/5)

        self.training = (self.x[:TRAINING_COUNT, :], self.y[:TRAINING_COUNT])
        self.testing  = (self.x[TRAINING_COUNT:, :], self.y[TRAINING_COUNT:])
