import numpy as np
import pandas as pd
import tensorflow as tf
import math
import random
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle 



            
def split_data():
    tdata=pd.read_csv("D:\\tianchi\\input.csv")

    train_data=tdata.iloc[:,:]
    train_x=np.array(train_data.iloc[:,:52])
    train_y=np.array(train_data.iloc[:,52])



    test_data=pd.read_csv("D:\\tianchi\\output.csv")
    test_x=np.array(test_data.iloc[:,:52])


    return train_x,train_y,test_x





train_x,train_y,test_x=split_data()

clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(128, 128,128), random_state=1,shuffle=True,max_iter=200)
clf.fit(train_x,train_y)
py=clf.predict(test_x)







np.savetxt("D:\\result.txt",py,fmt='%d')
print("done!")