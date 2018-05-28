import numpy as np
import pandas as pd
import tensorflow as tf
import math
import random
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle 


my_dict={'C':0,'D':1,'H':2,'S':3,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}

def trans_data():
    tdata=pd.read_csv("D:\\training.csv",header=None)
    data=[]
    len=tdata.shape[0]
    for i in range(len):
        e=np.zeros([53],dtype=np.int)
        for j in range(5):
            v1=my_dict[tdata.iloc[i,2*j]]
            v2=my_dict[tdata.iloc[i,2*j+1]]
            v=(v2-1)*4+v1
            e[v]=1
        e[52]=tdata.iloc[i,10]
        data.append(e)
        print(i*1.0/len)
    res=pd.DataFrame(data)
    res.to_csv("D:\\input.csv",header=False,index=False)
            
def split_data():
    tdata=pd.read_csv("D:\\tianchi\\input.csv")
    train_x=[]
    train_y=[]

    tdata=shuffle(tdata)

    train_data=tdata.iloc[:24000,:]


    for i in range(20):
        train_x.append(train_data.iloc[:20000,:52])
        train_y.append(train_data.iloc[:20000,52])
        train_data=shuffle(train_data)


    test_data=tdata.iloc[24000:,:]
    test_x=np.array(test_data.iloc[:,:52])
    test_y=np.array(test_data.iloc[:,52])

    return train_x,train_y,test_x,test_y


def solve1(train_x,train_y,test_x):
    py=[]
    for i in range(20):
        clf=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(128, 128),random_state=1,max_iter=300).fit(train_x[i],train_y[i])
        py.append(clf.predict(test_x))
    return py

def solve2(train_x,train_y,test_x):
    py=[]
    for i in range(20):
        clf=tree.DecisionTreeClassifier().fit(train_x[i],train_y[i])
        py.append(clf.predict(test_x))
    return py

#trans_data()
train_x,train_y,test_x,test_y=split_data()

sess=tf.Session()

res=[]

nn_py=solve1(train_x,train_y,test_x)
dt_py=solve2(train_x,train_y,test_x)

for i in range(test_x.shape[0]):
    t=[0,0,0,0,0,0,0,0,0,0]
    for j in range(20):
        temp=dt_py[j][i]
        t[temp]=t[temp]+1
        temp=nn_py[j][i]
        t[temp]=t[temp]+1
    '''
    flag=random.random()
    if(flag>1):
        pos=sess.run(tf.argmax(t))
        t[pos]=0
    '''
    res.append(sess.run(tf.argmax(t)))

cp=0
for i in range(test_y.shape[0]):
    if res[i]==test_y[i]:
        cp=cp+1
print(cp*1.0/test_y.shape[0])