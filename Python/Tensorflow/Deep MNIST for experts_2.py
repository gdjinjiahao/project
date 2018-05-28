import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)

def conv2d(x,w):
    '''
    strides[0] 控制 batch 
    strides[1] 控制 height 
    strides[2] 控制 width 
    strides[3] 控制 channels

    ·SAME：卷积输出与输入的尺寸相同。这里在计算如何跨越图像时，并不考虑滤波器的尺寸。选用该设置时，缺失的像素将用0填充，卷积核扫过的像素数将超过图像的实际像素数。 
·VALID：在计算卷积核如何在图像上跨越时，需要考虑滤波器的尺寸。这会使卷积核尽量不越过图像的边界。在某些情形下，可能边界也会被填充。 
在计算卷积时，最好能够考虑图像的尺寸，如果边界填充是必要的，则TensorFlow会有一些内置选项。在大多数比较简单的情形下，SAME都是一个不错的选择。当指定跨度参数后，如果输入和卷积核能够很好地工作，则推荐使用VALID。
    '''
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("ok")

x=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,shape=[None,10])

#构造第一层卷积层
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

x_image=tf.reshape(x,[-1,28,28,1])

h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_con1)

#构造第二层卷积层
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

#reshape函数相当于先把原shape变成一维，顺序类似数的先序遍历，之后再变为新的shape，填入顺序还是先序遍历
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#dropout
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.matmul(h_fc1_drop,w_fc2)+b_fc2

#train and evaluate
#tf.nn.softmax_cross_entropy_with_logits中集成了softmax函数和交叉熵计算
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
    #这里相当于每100次计算一下准确率，相当于测试或者预测，故概率必须为1，即全连接层全部神经元激活
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    #这里是正常训练，为防止过拟合，令一半神经元熄火
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))