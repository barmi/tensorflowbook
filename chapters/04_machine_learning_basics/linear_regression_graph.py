# Linear regression example in TF.

import tensorflow as tf
import numpy as np


def inference(X):
    return tf.add(tf.matmul(X, W, name='W_mul_X'), b, name='inference')


def evaluate(sess):
    print(sess.run(inference([[80., 25.]]))) # ~ 303
    print(sess.run(inference([[65., 25.]]))) # ~ 256


# weight_age
x_data = np.float32([
    [84, 46], [73, 20], [65, 52], [70, 30], [76, 57],
    [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
    [27, 24], [89, 31], [65, 52], [57, 23], [59, 60],
    [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
    [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]])
# blood_fat_content
y_data = [
    354, 190, 405, 263, 451,
    302, 288, 385, 402, 365,
    209, 290, 346, 254, 395,
    434, 220, 374, 308, 220,
    311, 181, 274, 303, 244]

graph = tf.Graph()

with graph.as_default():
    W = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    y = tf.add(tf.matmul(x_data, W, name="mul"), b, name="add")
    loss = tf.reduce_sum(tf.squared_difference(y, y_data), name="loss")
    train_op = tf.train.GradientDescentOptimizer(0.0000001).minimize(loss, name="GradientDescent")

# Launch the graph in a session, setup boilerplate
with tf.Session(graph=graph) as sess:

    writer = tf.summary.FileWriter('./linearReg', graph)

    tf.global_variables_initializer().run()

    '''
    def inference(X):
        return tf.add(tf.multiply(X, W, name='W_mul_X'), b, name='inference')


    def loss(X, Y):
        Y_predicted = inference(X)
        return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


    def train(total_loss):
        learning_rate = 0.0000001
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    '''

    # actual training loop
    training_steps = 10000
    for step in range(training_steps):
        result = sess.run(train_op)
        w_res = sess.run(W)
        if step % 100 == 0:
            print("%5d : W (%12.8f, %12.8f), b (%12.8f), loss: %12.8f" % (step, w_res[0], w_res[1], sess.run(b), sess.run(loss)))

    writer.flush()
    evaluate(sess)


