import warnings 
warnings.filterwarnings('ignore')
from dataset.ISEAR import Isear
from dataset.jigsaw import Jigsaw

data1 = Isear()

train_x,train_y,eval_x  , eval_y, test_x , test_y = data1.load()
print('shape of isear')
print(train_x.shape)
print(train_y.shape)
print(eval_x.shape)
print(eval_y.shape)
print(test_x.shape)
print(test_y.shape)

data2 = Jigsaw()

train_x,train_y,eval_x  , eval_y, test_x , test_y = data2.load()
print('shape of toxic comment data')
print(train_x.shape)
print(train_y.shape)
print(eval_x.shape)
print(eval_y.shape)
print(test_x.shape)
print(test_y.shape)

