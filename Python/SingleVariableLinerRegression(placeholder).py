import numpy as np
import tensorflow as tf

def createData(dataNum,w,b,sigma):  
    train_x = np.arange(dataNum)  
    train_y = w*train_x+b+np.random.randn()*sigma  
    #print train_x  
    #print train_y  
    return train_x,train_y  
  
  
def linerRegression(train_x,train_y,epoch=100000,rate = 0.000001):  
    train_x = np.array(train_x)  
    train_y = np.array(train_y)  
    n = train_x.shape[0]  
    x = tf.placeholder("float")  
    y = tf.placeholder("float")  
    w = tf.Variable(tf.random_normal([1])) # 生成随机权重  
    b = tf.Variable(tf.random_normal([1]))  
  
  
    pred = tf.add(tf.multiply(x,w),b)  
    loss = tf.reduce_sum(tf.pow(pred-y,2))  
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)  
  
    init = tf.initialize_all_variables()  
  
    sess = tf.Session()  
    sess.run(init)  

    for index in range(epoch):  
         
        sess.run(optimizer,feed_dict={x:train_x,y:train_y})  
    #只要用到涉及placeholder的变量，都需要设置feed_dict
    print ("loss is ",sess.run(loss,feed_dict={x:train_x,y:train_y})  )
    w =  sess.run(w)  
    b = sess.run(b)  
    return w,b  
  
def predictionTest(test_x,test_y,w,b):  
    W = tf.placeholder(tf.float32)  
    B = tf.placeholder(tf.float32)  
    X = tf.placeholder(tf.float32)  
    Y = tf.placeholder(tf.float32)  
    n = test_x.shape[0]  
    pred = tf.add(tf.mul(X,W),B)  
    loss = tf.reduce_mean(tf.pow(pred-Y,2))  
    sess = tf.Session()  
    loss = sess.run(loss,{X:test_x,Y:test_y,W:w,B:b})  
    return loss  

train_x,train_y = createData(50,2.0,7.0,1.0)  
w,b = linerRegression(train_x,train_y)  
print(w,b)
