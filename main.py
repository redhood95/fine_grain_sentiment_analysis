import warnings 
warnings.filterwarnings('ignore')
from dataset.ISEAR import Isear
from dataset.jigsaw import Jigsaw
from dataset.stanford import SST

dataset = 'jigsaw'


if dataset == 'isear':

    data1 = Isear()
    train_x,train_y,eval_x  , eval_y, test_x , test_y = data1.load()
elif dataset == 'jigsaw':
 
    data2 = Jigsaw()
    train_x,train_y,eval_x  , eval_y, test_x , test_y = data2.load()

elif dataset == 'stanford':  
    data3 = SST()
    train_x,train_y,eval_x  , eval_y, test_x , test_y = data3.load()
