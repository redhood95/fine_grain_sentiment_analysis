import pandas as pd 
import numpy as np
import regex as re

class Isear():
    
    def load(self,path):
        data = pd.read_csv(path,header = None)
        print(data[2][2])


