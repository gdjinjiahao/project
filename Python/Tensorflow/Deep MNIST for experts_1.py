import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("ok")

sess=tf.InteractiveSession()

x=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,shape=[None,10])

'''
初始化为0，准确率为约92%，初始化为正态分布，准确率约为88%
'''
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

#w=tf.Variable(tf.random_normal(shape=[784,10]))
#b=tf.Variable(tf.random_normal(shape=[10]))

sess.run(tf.global_variables_initializer())

y=tf.matmul(x,w)+b

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

for i in range(1000):
    batch=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1]})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))