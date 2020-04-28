import pandas as pd
import numpy as np
 

class Jigsaw:
    def load(self,path_2_data):
        train = pd.read_csv('data/jigsaw/train/train.csv')
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        y = train[list_classes].values
        list_sentences_train = train["comment_text"]
        max_features = 20000
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(list_sentences_train))
        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

        return x,y
