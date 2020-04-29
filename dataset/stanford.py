import pandas as pd 

# class SST:
#     def load(self):
#         df = pd.read_csv('data/stanford/SST.txt', sep='\t', header=None, names=['truth', 'text'])
#         print(df.head())


df = pd.read_csv('../data/stanford/SST.txt', sep='\t', header=None, names=['truth', 'text'])
print(len(df))

