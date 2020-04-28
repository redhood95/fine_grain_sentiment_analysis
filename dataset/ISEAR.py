import warnings 
warnings.filterwarnings('ignore')
import pandas as pd 
import numpy as np
import regex as re
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class Isear():
    
    def load(self):
        train = pd.read_csv('data/isear/ISEAR.csv')
        embed_size=0
        y = train['emotion'].values
        list_sentences_train = train["text"]
        max_features = 20000
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(list_sentences_train))
        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
        maxlen = 200
        X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        
        y = to_categorical(y)
        train_x , x , train_y , y = train_test_split(X_t , y , 
                                            test_size = 0.3 ,
                                            random_state = 324)

        eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
                                                            test_size = 0.5 , 
                                                            random_state = 324)

        return train_x,train_y,eval_x  , eval_y, test_x , test_y



    




        





