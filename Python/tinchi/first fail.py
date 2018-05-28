import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from sklearn import preprocessing

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

data=pd.read_csv("D:\\tianchi\\PINGAN-2018-train_demo.csv",header=0)


user_set=data["TERMINALNO"].drop_duplicates()

features=[]

tlabel=[]

ith=1

for i in user_set:

    feature1=data[(data.TERMINALNO==2) & ((data.CALLSTATE!=0) & (data.CALLSTATE!=4))].shape[0]*1.0/data[data.TERMINALNO==i].shape[0]

    feature2=data[(data.SPEED>0) & (data.TERMINALNO==i)].mean()[7]

    feature3=data[(data.SPEED>0) & (data.TERMINALNO==i)].max()[7]

    feature4=data[data.TERMINALNO==i].shape[0]

    features.append([feature1,feature2,feature3,feature4])

    temp=data[data.TERMINALNO==i].max()[9]
    if temp>0 :
        tlabel.append([0,1])
    else:
        tlabel.append([1,0])

    print(ith*1.0/len(user_set))
    ith=ith+1

features=preprocessing.scale(features)

train_x=np.array(features)

label=np.array(tlabel)







x=tf.placeholder(tf.float32,shape=[None,4])
y_=tf.placeholder(tf.float32,shape=[None,2])

w1=tf.Variable(tf.random_uniform([4,64], -1.0, 1.0),dtype=tf.float32)
b1=tf.Variable(tf.random_uniform([64], -1.0, 1.0),dtype=tf.float32)

y1=tf.nn.sigmoid(tf.matmul(x,w1)+b1)

w2=tf.Variable(tf.random_uniform([64,2], -1.0, 1.0),dtype=tf.float32)
b2=tf.Variable(tf.ones([2]),dtype=tf.float32)

y=tf.nn.sigmoid(tf.matmul(y1,w2)+b2)

loss=tf.reduce_mean(tf.square(y-y_))

train_step=tf.train.GradientDescentOptimizer(0.005).minimize(loss)

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())





gx=[]
gy=[]

#print(features)

for i in range(3000):
    sess.run(train_step,feed_dict={x:train_x[:90],y_:label[:90]})

    gx.append(i)
    gy.append(sess.run(loss,feed_dict={x:train_x[:90],y_:label[:90]}))

    #print(sess.run(loss,feed_dict={x:train_x[:90],y_:label[:90]}))
    if i%100==0:
        print((i+1)*1.0/3000)

'''
plt.plot(gx,gy)
plt.show()
'''


px=tf.placeholder(tf.float32,shape=[None,4])



py1=tf.nn.sigmoid(tf.matmul(px,w1)+b1)
py=tf.nn.sigmoid(tf.matmul(py1,w2)+b2)

print(sess.run(py,feed_dict={px:train_x[90:]}))




'''
x=[]

for i in range(len(user_set)):
    x.append(i)


plt.plot(x, feature1, linewidth=1, color="orange")
plt.plot(x, feature2, linewidth=1, color="red")
plt.plot(x, feature3, linewidth=1, color="green")
plt.plot(x, feature4, linewidth=1, color="blue")



# 设置图例
plt.legend(["Y","Z"], loc="upper right")
#plt.grid(True)
plt.show()
'''