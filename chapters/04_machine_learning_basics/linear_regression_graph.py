# Linear regression example in TF.

import tensorflow as tf
import numpy as np


def inference(X):
    return tf.add(tf.multiply(X, W, name='W_mul_X'), b, name='inference')


def loss(X, Y):
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def inputs():
    # Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight = [
        84, 73, 65,  70, 76,
        69, 63, 72,  79, 75,
        27, 89, 65,  57, 59,
        69, 60, 79,  75, 82,
        59, 67, 85,  55, 63 ]
    age = [
        46, 20, 52, 30, 57,
        25, 28, 36, 57, 44,
        24, 31, 52, 23, 60,
        48, 34, 51, 50, 34,
        46, 23, 37, 40, 30]
    blood_fat_content = [
        354, 190, 405, 263, 451,
        302, 288, 385, 402, 365,
        209, 290, 346, 254, 395,
        434, 220, 374, 308, 220,
        311, 181, 274, 303, 244]

    #return tf.to_float(weight), tf.to_float(age), tf.to_float(blood_fat_content)
    return np.array(weight, dtype=np.float32), np.array(age, dtype='f'), np.array(blood_fat_content, dtype='f')


def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    print(sess.run(inference([[80., 25.]]))) # ~ 303
    print(sess.run(inference([[65., 25.]]))) # ~ 256


graph = tf.Graph()

with graph.as_default():
    '''
    W = tf.placeholder(tf.float32, name="W_weights")
    '''
    W = tf.placeholder(tf.float32, name="W_weights")
    X = tf.placeholder(tf.float32, name="X_age")
    Y = tf.placeholder(tf.float32, name="Y_fat")
    b = tf.Variable(0., name="bias")

# Launch the graph in a session, setup boilerplate
with tf.Session(graph=graph) as sess:

    writer = tf.summary.FileWriter('./linearReg', graph)

    tf.global_variables_initializer().run()

    #w_data, x_data, y_data = inputs()
    w_data = [
        84, 73, 65, 70, 76,
        69, 63, 72, 79, 75,
        27, 89, 65, 57, 59,
        69, 60, 79, 75, 82,
        59, 67, 85, 55, 63]
    x_data = [
        46, 20, 52, 30, 57,
        25, 28, 36, 57, 44,
        24, 31, 52, 23, 60,
        48, 34, 51, 50, 34,
        46, 23, 37, 40, 30]
    y_data = [
        354, 190, 405, 263, 451,
        302, 288, 385, 402, 365,
        209, 290, 346, 254, 395,
        434, 220, 374, 308, 220,
        311, 181, 274, 303, 244]

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

    #total_loss = loss(X, Y)
    total_loss = tf.reduce_sum(tf.squared_difference(Y, W*X+b))
    #train_op = train(total_loss)
    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(total_loss)

    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        result = sess.run([train_op], feed_dict={X: x_data, W: w_data, Y: y_data})
        print(result)
        '''
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))
        '''

    writer.flush()

    evaluate(sess, X, Y)

    #coord.request_stop()
    #coord.join(threads)
    #sess.close()


