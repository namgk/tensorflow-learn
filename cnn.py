import tensorflow as tf
import argparse
import os

BATCH_SIZE = 100
DATA_DIR = '/vol/data'
LOGDIR = './train_model'
CHECKPOINT_EVERY = 100
NUM_STEPS = int(1e5)
CKPT_FILE = 'model.ckpt'
LEARNING_RATE = 1e-3
KEEP_PROB = 0.8
L2_REG = 0
EPSILON = 0.001
MOMENTUM = 0.9

def get_arguments():
  parser = argparse.ArgumentParser(description='ConvNet training')
  parser.add_argument('--store_metadata', type=bool, default=False,
                      help='Storing debug information for TensorBoard.')
  parser.add_argument('--restore_from', type=str, default=None,
                      help='Checkpoint file to restore model weights from.')
  return parser.parse_args()

args = get_arguments()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 1st conv layer: output 32 feature maps from 1 input channel, filter of size 5x5,
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

start_step = 0
if args.restore_from is not None:
  saver.restore(sess, args.restore_from)
  start_step = int(args.restore_from.split('step-')[1].split('-')[0])
  print('Model restored from ' + LOGDIR + '/' + args.restore_from)
  print('Start step: %d' % start_step)

if args.store_metadata:
  tf.summary.scalar("accuracy", accuracy)
  merged_summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(LOGDIR, graph=tf.get_default_graph())

max_accu = 0


for i in range(start_step, 500):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  
  if i%CHECKPOINT_EVERY == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

    if not os.path.exists(LOGDIR):
      os.makedirs(LOGDIR)

    checkpoint_path = os.path.join(LOGDIR, "model-step-%d-accuracy-%g.ckpt" % (i, train_accuracy))
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
    
    if train_accuracy > max_accu:
      max_accu = train_accuracy

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




