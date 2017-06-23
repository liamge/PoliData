import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class StripOutliersLength(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
        df_no_outliers = X[~(np.abs(X.lengths - X.lengths.mean()) > (2.5 * X.lengths.std()))]

        if y != None:
            return df_no_outliers['text'].values, df_no_outliers['label'].values

        return df_no_outliers['text'].values

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        from nltk import word_tokenize
        tokenized = []
        for text in X:
            tokenized.append(word_tokenize(text))

        return tokenized

class PadSequences(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        max_len = 0
        for text in X:
            if len(text) > max_len:
                max_len = len(text)

        for text in X:
            while len(text) < max_len:
                text.append('<PAD>')

        return X

class Indexizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if type(X[0]) == list:
            from itertools import chain
            self.vocab = set(list(chain(*X)))
            idx2w = enumerate(self.vocab)
            w2idx = {w: i for (i, w) in idx2w}
            del idx2w
            for l in X:
                for i in range(len(l)):
                    l[i] = w2idx[l[i]]
            return X
        else:
            tk = Tokenizer()
            X_tokenized = tk.transform(X)
            from itertools import chain
            self.vocab = set(list(chain(*X_tokenized)))
            idx2w = enumerate(self.vocab)
            w2idx = {w: i for (i, w) in idx2w}
            del idx2w
            for l in X_tokenized:
                for i in range(len(l)):
                    l[i] = w2idx[l[i]]
            return X_tokenized

class DataLoader:
    def __init__(self, csv_file, batch_size):
        self.batch_size = batch_size

        # Load and transform data
        df = pd.read_csv(csv_file)

        text_pipeline = Pipeline([
            ('strip_outliers', StripOutliersLength()),
            ('tokenizer', Tokenizer()),
            ('padder', PadSequences()),
            ('indexize', Indexizer())
        ])

        tk = Tokenizer()
        tokenized = tk.fit_transform(df.copy()['text'].values)

        vocab = []
        for line in tokenized:
            for w in line:
                if w not in vocab:
                    vocab.append(w)

        self.V = len(vocab)
        self.indexed = np.array(text_pipeline.fit_transform(df.copy()))
        self.labels = np.array(df[~(np.abs(df.lengths - df.lengths.mean()) > (2.5 * df.lengths.std()))]['label'].values)
        self.max_seq_len = self.indexed.shape[1]
        del tokenized, df, vocab

    def generate_batches(self, random_state=None, method='train'):
        if random_state is None:
            rnd = np.random.RandomState()
        elif isinstance(random_state, int):
            rnd = np.random.RandomState(random_state)
        else:
            rnd = random_state

        X_train, X_test, y_train, y_test = train_test_split(self.indexed, self.labels, random_state=random_state)

        if method == 'train':
            m = len(X_train)
            batch_size = self.batch_size if self.batch_size >= 1 else int(math.floor(m * self.batch_size))
            self.max_batches = int(math.floor(m / batch_size))
            cont = True

            while cont:
                random_indices = rnd.choice(np.arange(m), m, replace=False)
                for i in range(self.max_batches):
                    batch_indices = np.arange(i * batch_size, (i + 1) * batch_size)
                    indices = random_indices[batch_indices]
                    if self.labels is None:
                        yield X_train[indices]
                    else:
                        yield X_train[indices], y_train[indices]
                cont = False
        elif method == 'test':
            m = len(X_test)
            batch_size = self.batch_size if self.batch_size >= 1 else int(math.floor(m * self.batch_size))
            self.max_batches = int(math.floor(m / batch_size))
            cont = True

            while cont:
                random_indices = rnd.choice(np.arange(m), m, replace=False)
                for i in range(self.max_batches):
                    batch_indices = np.arange(i * batch_size, (i + 1) * batch_size)
                    indices = random_indices[batch_indices]
                    if self.labels is None:
                        yield X_test[indices]
                    else:
                        yield X_test[indices], y_test[indices]
                cont = False

class Config:
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 100
        self.embed_dim = 300
        self.direct_connections = True
        self.device = "/cpu:0"
        self.trainable = True
        self.hidden_dim = 1000