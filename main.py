import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create a placeholder tensor of dimension x X 784, x is of arbitrary length
x = tf.placeholder(tf.float32, [None, 784])

# weights
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))
# unnomarlized, could apply tf.nn.softmax on y
y = tf.matmul(x, W) + b

# lost between actual label (y_) and predicted label (y)
y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy Hy_(y) = sum(y_ * 1/log(y)) over element i 
# here softmax_cross_entropy_with_logits combines 
#   the calculation of softmax(y_)
#   and and the calculation of cross_entropy Hy_(y)
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))



