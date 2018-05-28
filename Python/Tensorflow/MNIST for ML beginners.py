import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("ok")

x=tf.placeholder(tf.float32,[None,784])

w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,w)+b)

y_=tf.placeholder(tf.float32,[None,10])

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

print(sess.run(correct_prediction,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print(sess.run(tf.cast(correct_prediction, tf.float32),feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))