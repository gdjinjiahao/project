import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from sklearn import preprocessing

def extract_data(data):
    tdata=np.array(data.loc[:,'class':'M'])
    np.random.shuffle(tdata)
    return tdata

def get_data(data):
    x=data[:,1:]
    t_y=[]
    for i in data[:,0]:
        tlist=[0,0,0]
        tlist[int(i)-1]=1
        t_y.append(tlist)
    y=np.array(t_y)
    return x,y


e_train_data=pd.read_csv("D:\\tianchi\\wine_train.csv")
e_test_data=pd.read_csv("D:\\tianchi\\wine_test.csv")

train_data=extract_data(e_train_data)
test_data=extract_data(e_test_data)

train_x,train_y=get_data(train_data)
test_x,test_y=get_data(test_data)

#标准化
train_x=preprocessing.scale(train_x)
train_y=preprocessing.scale(train_y)
test_x=preprocessing.scale(test_x)
test_y=preprocessing.scale(test_y)

x=tf.placeholder(tf.float32,shape=[None,13])
y_=tf.placeholder(tf.float32,shape=[None,3])

tx=tf.placeholder(tf.float32,shape=[None,13])
ty_=tf.placeholder(tf.float32,shape=[None,3])



w1=tf.Variable(tf.random_normal([13,256],mean=0,stddev=1,dtype=tf.float32))
b1=tf.Variable(tf.ones([256]))

y1=tf.nn.sigmoid(tf.matmul(x,w1)+b1)

w2=tf.Variable(tf.random_normal([256,3],mean=0,stddev=1,dtype=tf.float32))
b2=tf.Variable(tf.ones([3]))

y=tf.nn.sigmoid(tf.matmul(y1,w2)+b2)

loss=tf.reduce_mean(tf.square(y-y_))

train_step=tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

gx=[]
gy=[]

for i in range(3000):
    sess.run(train_step,feed_dict={x:train_x,y_:train_y})
    np.random.shuffle(train_data)
    train_x,train_y=get_data(train_data)

    gx.append(i)
    gy.append(sess.run(loss,feed_dict={x:train_x,y_:train_y}))


plt.plot(gx,gy)
plt.show()


#test
ty1=tf.nn.sigmoid(tf.matmul(tx,w1)+b1)
ty=tf.nn.sigmoid(tf.matmul(ty1,w2)+b2)

correct_prediction=tf.equal(tf.argmax(ty,1),tf.argmax(ty_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print("Accuracy:  ",sess.run(accuracy,feed_dict={tx:train_x,ty_:train_y}))
'''
print(sess.run(correct_prediction,feed_dict={tx:test_x,ty_:test_y}))
print(sess.run(ty_,feed_dict={ty_:test_y}))
'''