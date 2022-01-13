from gsdmm import MovieGroupProcess
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import re
import string
import gensim
from sklearn.base import TransformerMixin
import spacy

categories = ['alt.atheism', 'comp.graphics',
              'rec.sport.hockey', 'sci.crypt', 'talk.religion.misc']
newsgroups = fetch_20newsgroups(categories=categories)
y_true = newsgroups.target


class TextPreprocessor(TransformerMixin):
    def __init__(self, text_attribute):
        self.text_attribute = text_attribute

    def transform(self, X, *_):
        X_copy = X.copy()
        X_copy[self.text_attribute] = X_copy[self.text_attribute].apply(self._preprocess_text)
        return X_copy

    def _preprocess_text(self, text):
        return self._lemmatize(self._leave_letters_only(self._clean(text)))

    def _clean(self, text):
        bad_symbols = '!"#%&\'*+,-<=>?[\\]^_`{|}~'
        text_without_symbols = text.translate(str.maketrans('', '', bad_symbols))

        text_without_bad_words = ''
        for line in text_without_symbols.split('\n'):
            if not line.lower().startswith('from:') and not line.lower().endswith('writes:'):
                text_without_bad_words += line + '\n'

        clean_text = text_without_bad_words
        email_regex = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
        regexes_to_remove = [email_regex, r'Subject:', r'Re:']
        for r in regexes_to_remove:
            clean_text = re.sub(r, '', clean_text)

        return clean_text

    def _leave_letters_only(self, text):
        text_without_punctuation = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(re.findall("[a-zA-Z]+", text_without_punctuation))

    def _lemmatize(self, text):
        doc = nlp(text)
        words = [x.lemma_ for x in [y for y in doc if not y.is_stop and y.pos_ != 'PUNCT'
                                    and y.pos_ != 'PART' and y.pos_ != 'X']]
        return words

    def fit(self, *_):
        return self

nlp = spacy.load("en_core_web_sm")
df=pd.DataFrame({'text':newsgroups['data']})

text_preprocessor = TextPreprocessor(text_attribute='text')
df_preprocessed = text_preprocessor.transform(df)


docs=df_preprocessed.text.tolist()
dictionary = gensim.corpora.Dictionary(docs)

mgp = MovieGroupProcess(K=5, alpha=0.1, beta=0.1, n_iters=22)
y = mgp.fit(df_preprocessed.text.tolist(), len(dictionary))