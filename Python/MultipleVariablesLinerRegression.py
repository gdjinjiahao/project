import tensorflow as tf
import numpy as np

#模型结构定义
X = tf.placeholder(tf.float32, [None, 2])
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
Y = tf.placeholder(tf.float32, [None, 1])

# 成本函数 sum(sqr(y_-y))/n
cost = tf.reduce_mean(tf.square(Y-y))

# 用梯度下降训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)



#创建训练集
#产生50个2维数据
x_train=np.random.rand(50,2)
#定义实际的权重和偏差，注意数组形状
rw=np.array([[1.2],[2.7]])
rb=np.array([0.4])
#对于两个np.array对象，*运算和np.multiply都是要求相同shape的，操作为对应元素想乘，
#但这里我们要用的矩阵乘法，故使用np.dot，又但是，
#np.mat对象无法做feed_dict的映射，所在转化为np.array对象
#可用作映射的对象已知的有np.array,list
y_train=np.array(np.dot(x_train,rw)+rb)


for i in range(10000):
    sess.run(train_step, feed_dict={X: x_train, Y: y_train})
print("w0:%f" % sess.run(w[0]))
print("w1:%f" % sess.run(w[1]))
print("b:%f" % sess.run(b))
