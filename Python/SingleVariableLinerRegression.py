import numpy as np
import tensorflow as tf

def CreateData(dataNum):
    #生成100个随机数
    x_data=np.random.rand(dataNum).astype(np.float32)
    #真实函数
    y_data=x_data*0.5+0.14
    #返回训练集
    return x_data,y_data

def LinerRegression(train_x,train_y):
    #线性回归，先定义变量，即权重和偏值
    w=tf.Variable(tf.random_uniform([1],-1.0,1.0))
    b=tf.Variable(tf.zeros([1]))

    #定义线性函数
    y=w*train_x+b

    #定义误差
    loss=tf.reduce_mean(tf.square(y-train_y))

    #选择梯度优化，学习效率为0.5
    train=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    sess=tf.Session()
    init=tf.initialize_all_variables()
    sess.run(init)
    
    for step in range(500):
        sess.run(train)
        if step%100==0:
            print(sess.run(w),sess.run(b))

    return sess.run(w),sess.run(b),sess.run(loss)

train_x,train_y=CreateData(100)
w,b,loss=LinerRegression(train_x,train_y)
print("answer : ",w,b,loss)