import warnings 
warnings.filterwarnings('ignore')
import pandas as pd 
import numpy as np
import regex as re
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import json
import keras.utils as util


class Isear():
    
    def load(self,path):
        data = pd.read_csv(path,header = 0)
        y_target=data["emotion"]
        x_input=data["text"]

    def basic_preprocess(self,text):
        


