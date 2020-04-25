import pandas as pd
import numpy as np 

class Jigsaw:
    def load(self,path_2_data):
        train = pd.read_csv('data/jigsaw/train/train.csv')
        x = train["comment_text"].fillna("CVxTz").value
        y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

        return x,y
