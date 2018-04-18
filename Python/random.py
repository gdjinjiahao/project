import tensorflow as tf

a=tf.Variable(tf.random_normal([3,1]))

#[[-1.2412834 ]
# [ 0.13818105]
# [-0.01215874]]

a=tf.Variable(tf.random_normal([1]))

#[1.571668]



init = tf.initialize_all_variables()  
  
sess = tf.Session()  
sess.run(init)

print(sess.run(a)) 