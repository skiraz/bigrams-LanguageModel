# Note: test.txt file must be added to the folder directory

# imports
from pathlib import Path
from collections import Counter
import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


class LanguageModel():
    def __init__(self, file_path):
        self.file = str(Path(file_path).read_text(
            encoding='utf-8').replace("\n", ""))

    def Process(self, text=None):
        if not text:
            lem = WordNetLemmatizer()
            tokenizer = RegexpTokenizer(r'\w+')
            corpus = re.sub('[^A-Za-z0-9]+', ' ', self.file)
            corpus = corpus.lower()
            token = tokenizer.tokenize(corpus)
            self.corpus = [lem.lemmatize(w) for w in token]
        else:
            lem = WordNetLemmatizer()
            tokenizer = RegexpTokenizer(r'\w+')
            corpus = re.sub('[^A-Za-z0-9]+', ' ', text)
            corpus = corpus.lower()
            token = tokenizer.tokenize(corpus)
            text = [lem.lemmatize(w) for w in token]
            return text

    def Matrix(self):
        # get unique words
        uniq = set(self.corpus)

        # bigrams
        freq = Counter(nltk.bigrams(self.corpus))

        # creating matrix
        df = pd.DataFrame(columns=uniq, index=uniq)
        df = df.fillna(0)

        # filling matrix
        for i in freq.keys():
            df.loc[i[0], i[1]] = freq[i]

        # adding 1 for smoothing

        df = df + 1

        self.matrix = df

    def Run(self, sentence):
        last = self.Process(sentence)[-1]
        try:
            predicted = self.matrix.loc[last][self.matrix.loc[last] == max(
                self.matrix.loc[last])].index[0]
            print(predicted)
        except:
            print("UNKOWN")


model = LanguageModel(r'test.txt')
model.Process()
model.Matrix()
